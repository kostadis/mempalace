from mempalace.convo_miner import chunk_exchanges, CHUNK_SIZE

def test_chunking():
    # Simulate an exchange that is much larger than CHUNK_SIZE
    # CHUNK_SIZE is now 2000
    content = "> User says something very long\n" + ("A" * 5000) + "\nAI response\n" + ("B" * 5000)
    
    chunks = chunk_exchanges(content)
    print(f"Total chunks produced: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"Chunk {i} length: {len(c['content'])}")
        if len(c['content']) > 3000: # Sanity check
             print(f"!!! WARNING: Chunk {i} is too large: {len(c['content'])}")

if __name__ == "__main__":
    test_chunking()
