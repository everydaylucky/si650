``` Mermaid
graph TD
    %% Global Styles
    classDef stage1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef stage2 fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef stage3 fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef data fill:#ffffff,stroke:#333,stroke-dasharray: 5 5;
    classDef artifact fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10;

    %% Input
    Query[("User Query<br/>(Citation Context)")]:::artifact
    Corpus[("ArXiv Corpus<br/>(372 Source / 4504 Target)")]:::data

    %% Stage 1: Initial Retrieval
    subgraph S1 [Stage 1: Broad Retrieval - Top 1000]
        direction TB
        BM25[[BM25<br/>Sparse Retriever]]:::stage1
        TFIDF[[TF-IDF<br/>Sparse Retriever]]:::stage1
        SPECTER2[[SPECTER2<br/>Dense Retriever]]:::stage1
        
        Corpus -.-> BM25
        Corpus -.-> TFIDF
        Corpus -.-> SPECTER2
        
        Query --> BM25
        Query --> TFIDF
        Query --> SPECTER2
    end

    %% Merging
    Pool1[("Merged Candidates<br/>(Union of Stage 1)")]:::data
    BM25 --> Pool1
    TFIDF --> Pool1
    SPECTER2 --> Pool1

    %% Stage 2: Reranking
    subgraph S2 [Stage 2: Neural Reranking - Top 50]
        direction TB
        RRF[[Reciprocal Rank Fusion<br/>Combines S1 Scores]]:::stage2
        SciBERT[[SciBERT Bi-Encoder<br/>Fine-Tuned]]:::stage2
        
        Pool1 --> RRF
        RRF --> RRF_List[("RRF Ranked List")]:::data
        RRF_List --> SciBERT
    end

    Pool2[("Refined Candidates<br/>(Top 50)")]:::data
    SciBERT --> Pool2

    %% Stage 3: Final Ranking
    subgraph S3 [Stage 3: L2R and Cross-Encoder - Top 20]
        direction TB
        CE[[Cross-Encoder<br/>MS-MARCO / Fine-Tuned]]:::stage3
        FeatExt[[Feature Extraction<br/>17 Features]]:::stage3
        L2R[[LightGBM Ranker<br/>LambdaRank]]:::stage3
        
        %% Flow
        Pool2 --> CE
        Pool2 --> FeatExt
        
        %% Features
        CE -.->|Feature: CE Score| FeatExt
        S1 -.->|Feature: BM25/TF-IDF Scores| FeatExt
        S2 -.->|Feature: SciBERT Score| FeatExt
        
        FeatExt --> L2R
    end

    %% Output
    FinalResult[("Final Recommendations<br/>(Ordered List)")]:::artifact
    L2R --> FinalResult
```