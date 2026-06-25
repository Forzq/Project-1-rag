"""Учебная реализация quote-agent на LangGraph.

Файл намеренно содержит подробные комментарии. В рабочем проекте большую часть
таких объяснений обычно переносят в README и тесты, чтобы код оставался компактным.
"""

# hashlib нужен для SHA-256: из thread_id мы получаем стабильные ключи идемпотентности.
import hashlib
# date используется для проверки, успевает ли доставка к ISO-дедлайну.
from datetime import date
# Any обозначает значение произвольного типа.
# Literal ограничивает строку конкретным набором значений.
# TypedDict описывает структуру обычного словаря для статической проверки типов.
from typing import Any, Literal, TypedDict

# OpenRouter используется только внутри extractor; граф получает проверенный словарь.
from extractor import ExtractionError, extract_quote_details as extract_from_text
# MemorySaver сохраняет checkpoints в памяти текущего процесса.
from langgraph.checkpoint.memory import MemorySaver
# START и END являются специальными границами графа.
# StateGraph строит workflow, узлы которого читают и обновляют общее состояние.
from langgraph.graph import END, START, StateGraph

# Импортируем только разрешенные бизнес-инструменты из локального tools.py.
from tools import (
    # Вычисляет итоговую цену, не раскрывая агенту внутреннюю формулу.
    calc_price,
    # Создает черновик ответа клиенту.
    create_draft_reply,
    # Создает внутреннюю заметку для сотрудников.
    create_internal_note,
    # Возвращает клиента и разрешенные ему домены отправителей.
    crm_get_customer,
    # Проверяет остатки материала в ERP.
    erp_get_stock,
    # Рассчитывает стоимость доставки.
    shipping_rate,
)


# ---------------------------------------------------------------------------
# Состояние графа
# ---------------------------------------------------------------------------


# total=False означает, что на разных этапах некоторые ключи могут отсутствовать.
# Альтернатива: dataclass или Pydantic-модель. TypedDict проще для LangGraph и
# не создает объекты во время выполнения, но сам по себе не проверяет данные runtime.
class QuoteState(TypedDict, total=False):
    # Идентификатор цепочки писем; также используется для checkpoint и idempotency.
    thread_id: str
    # Полный адрес отправителя, например buyer@customer.com.
    sender_email: str
    # Список сообщений; каждое сообщение представлено словарем строк.
    messages: list[dict[str, str]]

    # Домен, извлеченный из sender_email.
    sender_domain: str
    # Карточка клиента, полученная из CRM.
    customer: dict[str, Any]
    # Результат проверки домена отправителя.
    sender_valid: bool

    # Флаг обнаружения признаков prompt injection.
    injection_detected: bool
    # Извлеченные параметры заказа.
    extracted_quote: dict[str, Any]
    # Источник извлечения: OpenRouter либо локальный regex fallback.
    extraction_source: Literal["openrouter", "regex"]
    # Обязательные параметры, которые не удалось найти.
    missing_fields: list[str]

    # Результат запроса остатков из ERP.
    stock: dict[str, Any]
    # Результат расчета цены.
    price: dict[str, Any]
    # Результат расчета доставки.
    shipping: dict[str, Any]

    # Стабильный ключ для создания черновика.
    draft_idempotency_key: str
    # Отдельный стабильный ключ для внутренней заметки.
    note_idempotency_key: str
    # Флаг успешного создания черновика в состоянии графа.
    draft_created: bool
    # Флаг успешного создания внутренней заметки.
    note_created: bool
    # Допустимы только два типа клиентского черновика.
    draft_kind: Literal["quote", "clarification"]

    # Требуется ли передать задачу человеку.
    escalation_required: bool
    # Причина передачи человеку или None, если причины нет.
    escalation_reason: str | None
    # Накопленные технические ошибки для диагностики.
    errors: list[str]


# ---------------------------------------------------------------------------
# Чистые вспомогательные функции
# ---------------------------------------------------------------------------


# Все поля, обязательные для расчета полноценной цены.
# Альтернатива: Enum или Pydantic-модель, особенно если у полей появятся правила.
QUOTE_FIELDS = ("size", "quantity", "material", "deadline", "country")

# Небольшой deny-list известных фраз prompt injection.
# Это только дополнительный сигнал. Основная защита состоит в том, что текст
# письма используется как данные и не может менять маршруты или системные правила.
# Альтернатива: отдельный классификатор, policy engine или LLM-классификация.
INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "developer message",
    "system prompt",
    "reveal pricing formula",
    "show pricing formula",
    "bypass validation",
    "do not call crm",
    "skip crm",
    "override business logic",
)


def _domain_from_email(email: str) -> str:
    """Вернуть нормализованный домен из email."""
    normalized_email = email.strip().lower()
    if normalized_email.count("@") != 1:
        raise ValueError("Email must contain exactly one @ character")

    local_part, domain = normalized_email.rsplit("@", 1)
    if not local_part or not domain or "." not in domain:
        raise ValueError("Email address is malformed")

    return domain


def _normalize_domains(domains: list[str] | set[str]) -> set[str]:
    """Нормализовать CRM-домены единообразно во всех узлах."""
    return {
        domain.strip().lower()
        for domain in domains
        if domain.strip()
    }


def _all_message_text(messages: list[dict[str, str]]) -> str:
    """Объединить все тела писем для общей security-проверки thread."""
    # Здесь намеренно читаются все сообщения текущего thread_id.
    # Это позволяет заметить injection даже в пересланном или чужом сообщении.
    return "\n\n".join(message.get("body", "") for message in messages)


def _customer_message_text(
    messages: list[dict[str, str]],
    allowed_domains: set[str],
) -> str:
    """Объединить только сообщения от разрешенных CRM-доменов."""
    # Функция принимает уже нормализованный set из _normalize_domains.
    normalized_domains = allowed_domains

    # В этот список попадут только тела сообщений доверенных отправителей.
    trusted_bodies: list[str] = []

    # Все сообщения уже относятся к одному заказу благодаря входному thread_id.
    for message in messages:
        # Поле from содержит адрес автора конкретного сообщения.
        message_sender = message.get("from", "")
        # Некорректный from считается недоверенным и пропускается.
        try:
            message_domain = _domain_from_email(message_sender)
        except ValueError:
            continue

        # Неизвестный домен не должен влиять на параметры заказа.
        if message_domain not in normalized_domains:
            continue

        # get не вызывает KeyError, если у сообщения отсутствует body.
        body = message.get("body", "")
        # Добавляем тело прошедшего проверку сообщения.
        trusted_bodies.append(body)

    # Два перевода строки сохраняют визуальную границу между письмами.
    # Альтернатива: фильтровать по точному sender_email. Это строже, но тогда
    # уточнение коллеги клиента с другого адреса того же CRM-домена потеряется.
    return "\n\n".join(trusted_bodies)


def _idempotency_key(kind: str, thread_id: str) -> str:
    """Создать одинаковый ключ для одинаковых kind и thread_id."""
    # kind разделяет пространство ключей: draft и note не получат один ключ.
    # encode переводит строку в bytes, потому что SHA-256 принимает байты.
    raw_key = f"{kind}:{thread_id}".encode("utf-8")
    # hexdigest возвращает строку из 64 hex-символов.
    # Альтернатива: UUIDv5. Случайный UUIDv4 здесь не подходит: при повторе он новый.
    return hashlib.sha256(raw_key).hexdigest()


def _missing_quote_fields(extracted_quote: dict[str, Any]) -> list[str]:
    """Вернуть список обязательных, но отсутствующих полей."""
    # List comprehension проходит по фиксированному QUOTE_FIELDS.
    # Поле отсутствует, если get вернул falsy-значение: None, "", 0 и т.д.
    return [field for field in QUOTE_FIELDS if not extracted_quote.get(field)]


def _clarifying_question(missing_fields: list[str]) -> str:
    """Собрать один вопрос сразу обо всех недостающих полях."""
    # join превращает ["size", "country"] в "size, country".
    readable_fields = ", ".join(missing_fields)
    # Один общий вопрос соблюдает ограничение "max 1 clarifying question".
    return f"Could you confirm the following quote details: {readable_fields}?"


# ---------------------------------------------------------------------------
# Узлы графа
# ---------------------------------------------------------------------------


def prepare_state(state: QuoteState) -> QuoteState:
    """Подготовить начальное состояние и ключи идемпотентности."""
    # Квадратные скобки намеренно требуют обязательный thread_id.
    # При отсутствии ключа будет KeyError вместо скрытого продолжения с плохими данными.
    thread_id = state["thread_id"]
    # Адрес также является обязательным входным полем.
    sender_email = state["sender_email"]

    try:
        sender_domain = _domain_from_email(sender_email)
        sender_error = None
    except ValueError as exc:
        sender_domain = ""
        sender_error = str(exc)

    # LangGraph объединит этот частичный словарь с общим состоянием.
    prepared_state: QuoteState = {
        # Извлекаем домен один раз, чтобы следующие узлы использовали готовое значение.
        "sender_domain": sender_domain,
        # Для одного thread_id всегда будет один и тот же draft-key.
        "draft_idempotency_key": _idempotency_key("draft", thread_id),
        # Для note используется другой kind, поэтому ключ отличается.
        "note_idempotency_key": _idempotency_key("note", thread_id),
        # Сохраняем True из восстановленного checkpoint, если draft уже создавался.
        "draft_created": bool(state.get("draft_created", False)),
        # Аналогично сохраняем статус внутренней заметки.
        "note_created": bool(state.get("note_created", False)),
        # При resume не стираем уже установленную эскалацию.
        "escalation_required": bool(
            state.get("escalation_required", False)
        ),
        # При resume сохраняем уже известную причину.
        "escalation_reason": state.get("escalation_reason"),
        # Копируем ошибки в новый list, чтобы не изменять входной список на месте.
        "errors": list(state.get("errors", [])),
    }

    if sender_error:
        prepared_state.update(
            {
                "sender_valid": False,
                "escalation_required": True,
                "escalation_reason": "Sender email is malformed",
                "errors": [
                    *prepared_state["errors"],
                    sender_error,
                ],
            }
        )

    return prepared_state


def validate_sender(state: QuoteState) -> QuoteState:
    """Проверить домен отправителя до обработки содержания письма."""
    if state.get("escalation_required") and not state.get("sender_domain"):
        return {"sender_valid": False}

    # try позволяет превратить ошибку внешнего CRM в контролируемую эскалацию.
    try:
        # CRM ищет клиента по адресу и возвращает allowed_sender_domains.
        customer = crm_get_customer(state["sender_email"])
    # Exception ловит timeout и другие ошибки wrapper-а.
    # В production лучше ловить конкретные ToolTimeout/ToolError.
    except Exception as exc:
        # При недоступной CRM продолжать расчет небезопасно.
        return {
            # Отправитель не считается подтвержденным.
            "sender_valid": False,
            # Запрос должен проверить человек.
            "escalation_required": True,
            # Сохраняем понятную бизнес-причину.
            "escalation_reason": "CRM lookup failed during sender validation",
            # * разворачивает старые ошибки и добавляет новую.
            "errors": [*state.get("errors", []), str(exc)],
        }

    # Одна helper-функция одинаково обрабатывает регистр и пробелы.
    allowed_domains = _normalize_domains(
        customer.get("allowed_sender_domains", [])
    )
    # True только если домен письма явно разрешен CRM.
    sender_valid = state["sender_domain"] in allowed_domains

    # Ветка для неизвестного или запрещенного домена.
    if not sender_valid:
        return {
            # Карточка полезна человеку при разборе инцидента.
            "customer": customer,
            # Явно фиксируем неуспешную проверку.
            "sender_valid": False,
            # Не разрешаем переход к извлечению и цене.
            "escalation_required": True,
            # Объясняем причину маршрутизации.
            "escalation_reason": "Sender domain is not allowed by CRM",
        }

    # Успешный результат содержит клиента и положительный флаг.
    return {
        "customer": customer,
        "sender_valid": True,
    }


def detect_injection(state: QuoteState) -> QuoteState:
    """Найти известные признаки prompt injection в тексте письма."""
    # Сначала объединяем письма, затем lower обеспечивает поиск без учета регистра.
    text = _all_message_text(state["messages"]).lower()
    # Собираем все сработавшие шаблоны, а не только первый.
    matched_patterns = [
        pattern
        for pattern in INJECTION_PATTERNS
        if pattern in text
    ]

    # Непустой список означает наличие хотя бы одного подозрительного сигнала.
    if matched_patterns:
        return {
            # Сохраняем отдельный security-флаг.
            "injection_detected": True,
            # Подозрительное письмо отдаем человеку.
            "escalation_required": True,
            # Причина не включает содержимое письма, чтобы не распространять его дальше.
            "escalation_reason": "Possible prompt injection detected",
            # Для диагностики записываем только имена сработавших шаблонов.
            "errors": [
                *state.get("errors", []),
                f"matched patterns: {matched_patterns}",
            ],
        }

    # Если совпадений нет, разрешаем routing-функции продолжить workflow.
    return {"injection_detected": False}


def extract_quote_details(state: QuoteState) -> QuoteState:
    """Извлечь параметры печати из недоверенного текста."""
    # Получаем единообразно нормализованные CRM-домены.
    allowed_domains = _normalize_domains(
        state["customer"].get("allowed_sender_domains", [])
    )
    # Для заказа используем только сообщения разрешенных CRM-доменов.
    customer_text = _customer_message_text(
        state["messages"],
        allowed_domains,
    )
    # OpenRouter включается только при наличии API key + model; иначе regex fallback.
    try:
        extraction = extract_from_text(
            customer_text,
            thread_id=state["thread_id"],
        )
    except ExtractionError as exc:
        return {
            "escalation_required": True,
            "escalation_reason": "Quote extraction failed",
            "errors": [*state.get("errors", []), str(exc)],
        }

    extracted_quote = extraction.details
    # Сравниваем результат с обязательным набором полей.
    missing_fields = _missing_quote_fields(extracted_quote)

    # Если отсутствует хотя бы одно поле, цена пока не рассчитывается.
    if missing_fields:
        return {
            # Сохраняем уже найденные значения.
            "extracted_quote": extracted_quote,
            "extraction_source": extraction.source,
            # Сохраняем полный список пропусков для одного общего вопроса.
            "missing_fields": missing_fields,
            # Следующий draft будет вопросом, а не коммерческим предложением.
            "draft_kind": "clarification",
        }

    # Полный набор данных готов к проверке ERP и расчетам.
    return {
        "extracted_quote": extracted_quote,
        "extraction_source": extraction.source,
        # Явно записываем пустой список.
        "missing_fields": [],
        # Следующий draft будет полноценной котировкой.
        "draft_kind": "quote",
    }


def fetch_stock(state: QuoteState) -> QuoteState:
    """Получить остатки после проверок отправителя и безопасности."""
    try:
        # В упрощенной модели material играет роль SKU.
        # Лучше иметь отдельный catalog lookup: material -> точный sku.
        stock = erp_get_stock(state["extracted_quote"]["material"])
    except Exception as exc:
        return {
            "escalation_required": True,
            "escalation_reason": "ERP stock lookup failed",
            "errors": [*state.get("errors", []), str(exc)],
        }

    # Записываем результат ERP в состояние для следующих узлов.
    return {"stock": stock}


def calculate_price(state: QuoteState) -> QuoteState:
    """Получить готовую цену, не раскрывая формулу."""
    # Короткое локальное имя делает следующие обращения читаемее.
    quote = state["extracted_quote"]

    try:
        # Передаем инструменту только нужные структурированные значения.
        price = calc_price(
            # В учебной версии material используется как sku.
            sku=quote["material"],
            # Количество уже преобразовано в int extractor-ом.
            quantity=quote["quantity"],
            # CRM customer_id позволяет применить разрешенные клиентские условия.
            customer_id=state["customer"].get("customer_id"),
        )
    except Exception as exc:
        return {
            "escalation_required": True,
            "escalation_reason": "Price calculation failed",
            "errors": [*state.get("errors", []), str(exc)],
        }

    # Граф получает только результат расчета, но не внутреннюю формулу.
    return {"price": price}


def calculate_shipping(state: QuoteState) -> QuoteState:
    """Рассчитать доставку для указанной страны."""
    # Повторно используем структурированный результат extraction.
    quote = state["extracted_quote"]

    try:
        # Инструмент получает страну назначения и выбранный уровень сервиса.
        shipping = shipping_rate(
            destination=quote["country"],
            service_level="standard",
        )
    except Exception as exc:
        return {
            "escalation_required": True,
            "escalation_reason": "Shipping calculation failed",
            "errors": [*state.get("errors", []), str(exc)],
        }

    # Сохраняем цену и признаки доступности доставки.
    return {"shipping": shipping}


def check_tool_conflicts(state: QuoteState) -> QuoteState:
    """Проверить бизнес-противоречия в результатах инструментов."""
    # get с {} защищает функцию от отсутствующего результата stock.
    stock = state.get("stock", {})
    # Аналогично получаем shipping.
    shipping = state.get("shipping", {})
    price = state.get("price", {})
    requested_quantity = state["extracted_quote"]["quantity"]

    if "total" not in price:
        return {
            "escalation_required": True,
            "escalation_reason": "Pricing tool returned no total",
        }

    if "cost" not in shipping:
        return {
            "escalation_required": True,
            "escalation_reason": "Shipping tool returned no cost",
        }

    # Используем `is False`, чтобы отличать явное False от отсутствующего значения.
    if stock.get("available") is False:
        return {
            "escalation_required": True,
            "escalation_reason": "Requested material is unavailable in stock",
        }

    available_quantity = stock.get(
        "available_quantity",
        stock.get("quantity"),
    )
    if (
        isinstance(available_quantity, int)
        and available_quantity < requested_quantity
    ):
        return {
            "escalation_required": True,
            "escalation_reason": (
                "ERP stock quantity is below the requested quantity"
            ),
        }

    # Невозможная доставка также требует ручного решения.
    if shipping.get("deliverable") is False:
        return {
            "escalation_required": True,
            "escalation_reason": (
                "Shipping tool says the destination is not deliverable"
            ),
        }

    requested_deadline = state["extracted_quote"].get("deadline")
    estimated_delivery = shipping.get("estimated_delivery_date")
    if isinstance(requested_deadline, str) and isinstance(
        estimated_delivery,
        str,
    ):
        try:
            deadline_date = date.fromisoformat(requested_deadline)
            delivery_date = date.fromisoformat(estimated_delivery)
        except ValueError:
            # Non-ISO dates cannot be compared safely and stay informational.
            pass
        else:
            if delivery_date > deadline_date:
                return {
                    "escalation_required": True,
                    "escalation_reason": (
                        "Estimated delivery is later than the deadline"
                    ),
                }

    # Пустое обновление означает, что конфликтов не найдено.
    # Можно дополнительно сравнить stock quantity, deadline и estimated delivery.
    return {}


def create_draft(state: QuoteState) -> QuoteState:
    """Создать не более одного клиентского черновика для thread."""
    # Первый барьер от повторного side effect после восстановления checkpoint.
    if state.get("draft_created"):
        # Ничего не обновляем и не вызываем внешний инструмент повторно.
        return {}

    # Для неполных данных формируем один уточняющий вопрос.
    if state["draft_kind"] == "clarification":
        body = _clarifying_question(state["missing_fields"])
    # Иначе данных достаточно для draft quote.
    else:
        # Локальные имена сокращают повторяющиеся обращения к state.
        quote = state["extracted_quote"]
        price = state["price"]
        shipping = state["shipping"]
        # Склеиваем безопасный шаблон только из структурированных результатов.
        # Pricing formula здесь отсутствует: клиент видит только итоговые значения.
        body = (
            "Thanks for your request. Here is a draft quote summary:\n\n"
            f"- Size: {quote['size']}\n"
            f"- Quantity: {quote['quantity']}\n"
            f"- Material: {quote['material']}\n"
            f"- Requested deadline: {quote['deadline']}\n"
            f"- Delivery country: {quote['country']}\n"
            f"- Estimated item total: {price.get('total')} "
            f"{price.get('currency', '')}\n"
            f"- Estimated shipping: {shipping.get('cost')} "
            f"{shipping.get('currency', '')}\n\n"
            "Please review the details and confirm if you would like to proceed."
        )

    # Это внешний side effect: инструмент реально создает draft.
    try:
        create_draft_reply(
            # thread_id позволяет связать side effect с текущей перепиской.
            thread_id=state["thread_id"],
            # CRM id связывает draft с клиентом.
            customer_id=state["customer"].get("customer_id", ""),
            # body сформирован нашим шаблоном, а не командами из письма.
            body=body,
            # Один thread всегда передает один ключ.
            idempotency_key=state["draft_idempotency_key"],
        )
    except Exception as exc:
        return {
            "escalation_required": True,
            "escalation_reason": (
                "Draft creation outcome requires manual verification"
            ),
            "errors": [*state.get("errors", []), str(exc)],
        }

    # Флаг записывается только после успешного возвращения инструмента.
    # Важно: при падении между side effect и checkpoint возможен дубль.
    # Надежное решение требует, чтобы сам wrapper/API атомарно сохранял ключ.
    return {"draft_created": True}


def create_note(state: QuoteState) -> QuoteState:
    """Создать не более одной внутренней заметки для thread."""
    # Не повторяем вызов, если checkpoint уже содержит успешный флаг.
    if state.get("note_created"):
        return {}

    # При эскалации сотруднику нужна конкретная причина.
    if state.get("escalation_required"):
        body = (
            "Escalated quote thread. "
            f"Reason: {state.get('escalation_reason')}"
        )
    # При clarification записываем, каких данных не хватило.
    elif state.get("draft_kind") == "clarification":
        body = (
            "Clarification draft created. "
            f"Missing fields: {state['missing_fields']}"
        )
    # Иначе фиксируем штатное создание quote draft.
    else:
        body = (
            "Quote draft created from validated sender "
            "and internal pricing tools."
        )

    # Создание note является вторым внешним side effect.
    try:
        create_internal_note(
            # Note можно привязать к thread даже при ранней ошибке CRM.
            thread_id=state["thread_id"],
            # При ранней эскалации customer может отсутствовать.
            customer_id=state.get("customer", {}).get("customer_id"),
            # Текст заметки сформирован бизнес-логикой.
            body=body,
            # Note имеет собственный детерминированный ключ.
            idempotency_key=state["note_idempotency_key"],
        )
    except Exception as exc:
        return {
            "note_created": False,
            "escalation_required": True,
            "escalation_reason": (
                state.get("escalation_reason")
                or "Internal note creation requires manual verification"
            ),
            "errors": [*state.get("errors", []), str(exc)],
        }

    # Отмечаем успешное создание заметки.
    return {"note_created": True}


def escalate(state: QuoteState) -> QuoteState:
    """Остановить автоматическую обработку и передать запрос человеку."""
    return {
        # Даже если предыдущий узел не поставил флаг, этот узел поставит.
        "escalation_required": True,
        # Сохраняем конкретную причину либо используем безопасное значение по умолчанию.
        "escalation_reason": (
            state.get("escalation_reason") or "Manual review required"
        ),
    }


# ---------------------------------------------------------------------------
# Функции маршрутизации
# ---------------------------------------------------------------------------


def route_after_sender_validation(
    state: QuoteState,
) -> Literal["continue", "escalate"]:
    """Выбрать путь после CRM-проверки."""
    # Любая ошибка CRM или невалидный домен ведут к человеку.
    if state.get("escalation_required") or not state.get("sender_valid"):
        return "escalate"
    # Только явно подтвержденный отправитель продолжает обработку.
    return "continue"


def route_after_injection_check(
    state: QuoteState,
) -> Literal["continue", "escalate"]:
    """Выбрать путь после security-проверки."""
    # Проверяем общий escalation-флаг и специальный injection-флаг.
    if state.get("escalation_required") or state.get("injection_detected"):
        return "escalate"
    # Без сигналов риска можно извлекать параметры заказа.
    return "continue"


def route_after_extraction(
    state: QuoteState,
) -> Literal["quote", "clarification", "escalate"]:
    """Выбрать расчет либо единственный уточняющий вопрос."""
    if state.get("escalation_required"):
        return "escalate"
    # Непустой список missing_fields требует clarification.
    if state.get("missing_fields"):
        return "clarification"
    # Полный набор полей отправляется к внутренним инструментам.
    return "quote"


def route_after_tool_step(
    state: QuoteState,
) -> Literal["continue", "escalate"]:
    """Единообразно обработать результат любого внутреннего инструмента."""
    # Каждый tool-node при ошибке устанавливает один общий флаг.
    if state.get("escalation_required"):
        return "escalate"
    # Без ошибки workflow идет к следующему шагу.
    return "continue"


# ---------------------------------------------------------------------------
# Сборка графа
# ---------------------------------------------------------------------------


def build_graph():
    """Зарегистрировать узлы, связи и checkpoint-хранилище."""
    # Создаем builder и сообщаем ему тип общего состояния.
    graph = StateGraph(QuoteState)

    # Каждая строка связывает строковое имя узла с Python-функцией.
    graph.add_node("prepare_state", prepare_state)
    graph.add_node("validate_sender", validate_sender)
    graph.add_node("detect_injection", detect_injection)
    graph.add_node("extract_quote_details", extract_quote_details)
    graph.add_node("fetch_stock", fetch_stock)
    graph.add_node("calculate_price", calculate_price)
    graph.add_node("calculate_shipping", calculate_shipping)
    graph.add_node("check_tool_conflicts", check_tool_conflicts)
    graph.add_node("create_draft", create_draft)
    graph.add_node("create_note", create_note)
    graph.add_node("escalate", escalate)

    # START всегда ведет к подготовке состояния.
    graph.add_edge(START, "prepare_state")
    # После подготовки обязательно выполняется CRM-проверка.
    graph.add_edge("prepare_state", "validate_sender")

    # Conditional edge вызывает routing-функцию после validate_sender.
    graph.add_conditional_edges(
        # Имя узла, после которого выбирается маршрут.
        "validate_sender",
        # Функция вернет строковую метку continue или escalate.
        route_after_sender_validation,
        {
            # Метка continue переводит к проверке injection.
            "continue": "detect_injection",
            # Метка escalate переводит к ручной обработке.
            "escalate": "escalate",
        },
    )

    # После security-проверки либо извлекаем данные, либо эскалируем.
    graph.add_conditional_edges(
        "detect_injection",
        route_after_injection_check,
        {
            "continue": "extract_quote_details",
            "escalate": "escalate",
        },
    )

    # Полные данные идут к ERP; неполные сразу формируют один вопрос.
    graph.add_conditional_edges(
        "extract_quote_details",
        route_after_extraction,
        {
            "quote": "fetch_stock",
            "clarification": "create_draft",
            "escalate": "escalate",
        },
    )

    # После ERP либо считаем цену, либо прекращаем автоматический путь.
    graph.add_conditional_edges(
        "fetch_stock",
        route_after_tool_step,
        {
            "continue": "calculate_price",
            "escalate": "escalate",
        },
    )

    # После цены либо считаем доставку, либо эскалируем.
    graph.add_conditional_edges(
        "calculate_price",
        route_after_tool_step,
        {
            "continue": "calculate_shipping",
            "escalate": "escalate",
        },
    )

    # После доставки проверяем совместимость результатов.
    graph.add_conditional_edges(
        "calculate_shipping",
        route_after_tool_step,
        {
            "continue": "check_tool_conflicts",
            "escalate": "escalate",
        },
    )

    # Без конфликтов создаем draft; при конфликте нужен человек.
    graph.add_conditional_edges(
        "check_tool_conflicts",
        route_after_tool_step,
        {
            "continue": "create_draft",
            "escalate": "escalate",
        },
    )


    # После клиентского draft всегда создается внутренняя note.
    graph.add_edge("create_draft", "create_note")
    # После note workflow завершается.
    graph.add_edge("create_note", END)

    # Эскалация не создает клиентский ответ, а только внутреннюю note.
    graph.add_edge("escalate", "create_note")

    # Создаем in-memory checkpoint-хранилище.
    # Для production лучше durable saver, например PostgreSQL.
    checkpointer = MemorySaver()
    # compile валидирует структуру и возвращает исполняемый graph.
    return graph.compile(checkpointer=checkpointer)


# Собираем один объект при импорте модуля, чтобы его могли использовать тесты.
quote_graph = build_graph()


def run_quote_agent(input_data: QuoteState) -> QuoteState:
    """Запустить граф вручную с checkpoint-id текущей email-цепочки."""
    # LangGraph использует configurable.thread_id как ключ checkpoint-истории.
    config = {
        "configurable": {
            "thread_id": input_data["thread_id"],
        }
    }
    # invoke запускает workflow синхронно и возвращает итоговое состояние.
    # Альтернатива: ainvoke для async-приложения или stream для показа шагов.
    return quote_graph.invoke(input_data, config=config)
