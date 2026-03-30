"""
Simple manual test for /v1/sessions/prune endpoint.

This is a minimal test that can be run quickly to verify the endpoint works.

Usage:
    1. Start SGLang server:
       python -m sglang.launch_server --port 8080 --enable-semantic-pruning
    
    2. Run this test:
       python manual_test_prune_endpoint.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:8080"


def test_prune_endpoint():
    """Quick test of the prune endpoint."""
    print("Testing /v1/sessions/prune endpoint")
    print("=" * 50)

    # 1. Check server health
    print("\n1. Checking server health...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            print("   ✓ Server is running")
        else:
            print(f"   ✗ Server returned {r.status_code}")
            return
    except Exception as e:
        print(f"   ✗ Cannot connect: {e}")
        return

    # 2. Create a session with some conversation
    print("\n2. Creating session...")
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": 10,
            "semantic_event": "start"
        },
        timeout=30
    )

    if r.status_code != 200:
        print(f"   ✗ Failed: {r.status_code}")
        return

    data = r.json()
    session_id = data.get("metadata", {}).get("session_id")
    print(f"   ✓ Session created: {session_id}")

    # 3. Build up some conversation history
    print("\n3. Building conversation history...")
    conversation = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2=4"},
        {"role": "user", "content": "What about 3+3?"},
        {"role": "assistant", "content": "3+3=6"},
    ]

    for i in range(2):
        r = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": conversation,
                "max_tokens": 10,
                "session_params": {"id": session_id}
            },
            timeout=30
        )
        if r.status_code == 200:
            conversation.append({
                "role": "user",
                "content": f"Question {i+3}?"
            })
            conversation.append({
                "role": "assistant",
                "content": r.json()["choices"][0]["message"]["content"]
            })
    print("   ✓ History built")

    # 4. Check KV cache before pruning
    print("\n4. Checking KV cache before pruning...")
    r = requests.get(f"{BASE_URL}/v1/kv_cache", params={"include": "tree"}, timeout=5)
    if r.status_code == 200:
        tree_before = r.json()["kv_cache_state"][0]["tree"]["total_size"]
        print(f"   Tree size: {tree_before} nodes")
    else:
        tree_before = 0
        print("   Could not get tree size")

    # 5. Call prune endpoint
    print("\n5. Calling /v1/sessions/prune...")
    
    # Use Qwen chat template format for proper prefix matching
    def format_qwen_prompt(messages):
        prompt = ""
        for i, msg in enumerate(messages):
            role = msg['role']
            content = msg['content']
            if i == 0 and role != 'system':
                prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return prompt
    
    full_prompt = format_qwen_prompt(conversation)

    start = time.time()
    r = requests.post(
        f"{BASE_URL}/v1/sessions/prune",
        json={
            "session_id": session_id,
            "prompt": full_prompt,
            "prune_mode": "subtree"
        },
        timeout=10
    )
    latency = (time.time() - start) * 1000

    if r.status_code == 200:
        result = r.json()
        print(f"   ✓ Success!")
        print(f"   Latency: {latency:.2f}ms")
        print(f"   Matched prefix: {result.get('matched_prefix_length', 0)} tokens")
        print(f"   Message: {result.get('message', 'N/A')}")
    else:
        print(f"   ✗ Failed: {r.status_code} - {r.text[:100]}")
        return

    # 6. Wait for pruning to take effect (via idle processing)
    print("\n6. Waiting for pruning to take effect via idle processing...")
    print("   (No trigger requests needed - pruning happens during idle time)")
    
    # Wait for the scheduler to go idle and process pending prunes
    # This typically happens within a few hundred milliseconds
    time.sleep(1.0)

    r = requests.get(f"{BASE_URL}/v1/kv_cache", params={"include": "tree"}, timeout=5)
    if r.status_code == 200:
        tree_after = r.json()["kv_cache_state"][0]["tree"]["total_size"]
        reduction = tree_before - tree_after
        print(f"   Tree size: {tree_after} nodes (reduction: {reduction})")
        if reduction > 0:
            print("   ✓ Pruning worked!")
        else:
            print("   ⚠ No reduction (may need different timing)")
            print("   Note: Check server logs for [PRUNE-PENDING] messages")
    else:
        print("   Could not get tree size")

    print("\n" + "=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    test_prune_endpoint()
