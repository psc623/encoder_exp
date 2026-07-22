from encoderbench.zero_shot import _contains_final_answer, _shard_rows


def test_final_answer_stopping_marker():
    assert _contains_final_answer("Reasoning\nFinal Answer: AD", "ad")
    assert _contains_final_answer("final answer - cn", "ad")
    assert _contains_final_answer("Final Answer: SCZ", "scz")
    assert not _contains_final_answer("AD remains possible", "ad")
    assert not _contains_final_answer("Final Answer: SCZ", "ad")


def test_shards_are_disjoint_and_preserve_global_order():
    rows = [{"file_id": str(index)} for index in range(11)]
    shards = [_shard_rows(rows, 4, index) for index in range(4)]
    assert [[row["file_id"] for row in shard] for shard in shards] == [
        ["0", "4", "8"], ["1", "5", "9"], ["2", "6", "10"], ["3", "7"]
    ]
    assert sorted(int(row["file_id"]) for shard in shards for row in shard) == list(range(11))
