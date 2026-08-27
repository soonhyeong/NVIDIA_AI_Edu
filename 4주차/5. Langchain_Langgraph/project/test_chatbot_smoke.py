"""외부 API 호출 없이 실행 가능한 핵심 로직 smoke test."""

from langchain_core.documents import Document

from project.Chat_Bot import calculate_income_tax_amount, split_law_articles


def test_income_tax_brackets() -> None:
    assert calculate_income_tax_amount(0)["calculated_tax"] == 0
    assert calculate_income_tax_amount(14_000_000)["calculated_tax"] == 840_000
    assert calculate_income_tax_amount(50_000_000)["calculated_tax"] == 6_240_000
    assert calculate_income_tax_amount(60_000_000)["calculated_tax"] == 8_640_000
    assert calculate_income_tax_amount(1_100_000_000)["calculated_tax"] == 429_060_000


def test_split_law_articles_preserves_metadata() -> None:
    source = Document(
        page_content="제1조(목적) 첫 번째 조문입니다.\n제1조의2(정의) 두 번째 조문입니다.",
        metadata={"source": "sample.docx"},
    )
    articles = split_law_articles(source)
    assert [item.metadata["article_number"] for item in articles] == ["제1조", "제1조의2"]
    assert articles[1].metadata["article_title"] == "정의"
    assert articles[0].metadata["source"] == "sample.docx"
