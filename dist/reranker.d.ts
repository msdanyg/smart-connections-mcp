export interface RerankCandidate {
    path: string;
    text: string;
}
export interface RerankResult {
    path: string;
    score: number;
}
/**
 * Rerank candidates by relevance to query. Returns sorted desc by cross-encoder score.
 * text는 호출자가 노트 본문 발췌(≤~2000자 권장; tokenizer가 512 truncate)를 전달.
 */
export declare function rerank(query: string, candidates: RerankCandidate[]): Promise<RerankResult[]>;
export declare function rerankerReady(): boolean;
//# sourceMappingURL=reranker.d.ts.map