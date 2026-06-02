/**
 * Cross-encoder reranker — onnx-community/bge-reranker-v2-m3-ONNX (multilingual, q8).
 * 2026-06-02 도입: systematic eval에서 union(dense∪bm25)→rerank가 양 쿼리타입 최선
 * (semantic R@1 98% / exact-match 92%). dense-only/RRF 대비 우월.
 * lazy-load (gte-embedder와 동일 패턴, MCP init timeout 회피).
 */
import { AutoTokenizer, AutoModelForSequenceClassification } from '@huggingface/transformers';
const RERANK_MODEL = 'onnx-community/bge-reranker-v2-m3-ONNX';
let tokenizer = null;
let model = null;
let loadPromise = null;
async function ensureLoaded() {
    if (model && tokenizer)
        return;
    if (!loadPromise) {
        loadPromise = (async () => {
            console.error('Loading bge-reranker-v2-m3 (first run downloads ~600MB)...');
            tokenizer = await AutoTokenizer.from_pretrained(RERANK_MODEL);
            model = await AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, { dtype: 'q8' });
            console.error('bge-reranker-v2-m3 loaded');
        })();
    }
    await loadPromise;
}
/**
 * Rerank candidates by relevance to query. Returns sorted desc by cross-encoder score.
 * text는 호출자가 노트 본문 발췌(≤~2000자 권장; tokenizer가 512 truncate)를 전달.
 */
export async function rerank(query, candidates) {
    if (candidates.length === 0)
        return [];
    await ensureLoaded();
    const scored = [];
    for (const c of candidates) {
        const inputs = await tokenizer([query], { text_pair: [c.text], padding: true, truncation: true, max_length: 512 });
        const { logits } = await model(inputs);
        scored.push({ path: c.path, score: Number(logits.data[0]) });
    }
    return scored.sort((a, b) => b.score - a.score);
}
export function rerankerReady() { return !!(model && tokenizer); }
//# sourceMappingURL=reranker.js.map