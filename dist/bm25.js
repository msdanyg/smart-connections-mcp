/**
 * BM25 lexical index — exact-match(저자명/약어/한글 고유명사) 보완.
 * 2026-06-02 도입: systematic eval에서 dense는 exact-match R@1 68%, BM25는 97%.
 * dense의 lexical blur를 보완. union(dense∪bm25)→reranker 파이프라인의 lexical leg.
 * 순수 in-memory, 인덱스 파일 없음 (load 시 노트 텍스트로 build → 인덱스버전 변경 0).
 */
const STOP = new Set(['연구', '분석', '결과', 'the', 'a', 'an', 'of', 'and', 'to', 'in', 'is', 'for', 'with', 'on']);
export function tokenize(s) {
    return s.toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '')
        .split(/[^a-z0-9가-힣]+/).filter(t => t.length > 0 && !STOP.has(t));
}
export class BM25Index {
    docToks = [];
    paths = [];
    df = new Map();
    avgdl = 0;
    k1 = 1.5;
    b = 0.75;
    /** docs: [{path, text}] — text는 title+본문 발췌 */
    build(docs) {
        this.paths = docs.map(d => d.path);
        this.docToks = docs.map(d => tokenize(d.text));
        this.df.clear();
        for (const toks of this.docToks) {
            for (const t of new Set(toks))
                this.df.set(t, (this.df.get(t) || 0) + 1);
        }
        const total = this.docToks.reduce((a, d) => a + d.length, 0);
        this.avgdl = total / (this.docToks.length || 1);
    }
    get size() { return this.paths.length; }
    /** query에 대해 상위 k개 path 반환 (BM25 score desc, score>0만) */
    topK(query, k) {
        const qt = tokenize(query);
        const N = this.docToks.length;
        const scores = [];
        for (let i = 0; i < N; i++) {
            const d = this.docToks[i];
            const tf = new Map();
            for (const t of d)
                tf.set(t, (tf.get(t) || 0) + 1);
            let s = 0;
            for (const t of qt) {
                const df = this.df.get(t) || 0;
                if (!df)
                    continue;
                const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
                const f = tf.get(t) || 0;
                s += idf * (f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + this.b * d.length / this.avgdl));
            }
            if (s > 0)
                scores.push({ path: this.paths[i], score: s });
        }
        return scores.sort((a, b) => b.score - a.score).slice(0, k);
    }
}
//# sourceMappingURL=bm25.js.map