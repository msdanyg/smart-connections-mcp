/**
 * BM25 lexical index — exact-match(저자명/약어/한글 고유명사) 보완.
 * 2026-06-02 도입: systematic eval에서 dense는 exact-match R@1 68%, BM25는 97%.
 * dense의 lexical blur를 보완. union(dense∪bm25)→reranker 파이프라인의 lexical leg.
 * 순수 in-memory, 인덱스 파일 없음 (load 시 노트 텍스트로 build → 인덱스버전 변경 0).
 */
export declare function tokenize(s: string): string[];
export declare class BM25Index {
    private docToks;
    private paths;
    private df;
    private avgdl;
    private k1;
    private b;
    /** docs: [{path, text}] — text는 title+본문 발췌 */
    build(docs: {
        path: string;
        text: string;
    }[]): void;
    get size(): number;
    /** query에 대해 상위 k개 path 반환 (BM25 score desc, score>0만) */
    topK(query: string, k: number): {
        path: string;
        score: number;
    }[];
}
//# sourceMappingURL=bm25.d.ts.map