from project_rag.chunker.chunker import page_for_offset, stratagem_chunk_text

PART_STARTS = [0, 10, 25]
PART_PAGES = [5, 5, 6]


def test_offset_inside_first_part():
    assert page_for_offset(PART_STARTS, PART_PAGES, 3) == 5


def test_offset_on_part_boundary_belongs_to_new_part():
    assert page_for_offset(PART_STARTS, PART_PAGES, 25) == 6


def test_offset_inside_last_part():
    assert page_for_offset(PART_STARTS, PART_PAGES, 40) == 6


def test_stratagem_text_starts_with_the_stratagem_name():
    text = stratagem_chunk_text("FIRE OVERWATCH", ["CORE - STRATEGIC PLOY STRATAGEM", "WHEN: ..."], "1CP")
    assert text.splitlines()[0] == "FIRE OVERWATCH"
    assert text.endswith("[CP cost: 1CP]")


def test_stratagem_text_without_known_cost_has_no_cost_line():
    assert "CP cost" not in stratagem_chunk_text("X", ["line"], None)
