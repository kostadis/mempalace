import sys
from mempalace.convo_miner import _embed_prepared_convo, _PreparedConvo, _ConvoBatch

class MockEF:
    def __call__(self, docs):
        return [[0.1]*768 for _ in docs]

def test():
    ef = MockEF()
    
    # Test 1: A single chunk that is actually quite large (e.g. 10,000 chars)
    # 10,000 chars is roughly 2,500 tokens. This should trigger the error if we used the real EF.
    large_content = "a" * 10000
    prepared = _PreparedConvo(
        source_file="test.txt", 
        chunks=[{"content": large_content, "chunk_index": 0}], 
        room="test", 
        extract_mode="exchange", 
        batches=[_ConvoBatch(documents=[large_content], ids=["id1"], metadatas=[{}], rooms=["test"])]
    )
    
    print("Testing 10,000 chars (approx 2,500 tokens) with MockEF...")
    res = _embed_prepared_convo(prepared, ef)
    print(f"Success: Got {len(res)} embedding batch(es)")

if __name__ == "__main__":
    test()
