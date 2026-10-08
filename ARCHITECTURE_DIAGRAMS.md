# MindCare System Architecture & Pipelines

This document contains the structural diagrams for the MindCare AI platform. You can use these diagrams for your Final Year Project (FYP) report, presentation, or to answer committee questions regarding data flow and system design. You can view these diagrams by pasting the code blocks into [Mermaid Live Editor](https://mermaid.live).

## 1. High-Level System Architecture

This diagram shows how the entire system connects together, from the user interface down to the backend machine learning models.

```mermaid
graph TD
    %% Define styles
    classDef frontend fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:white;
    classDef backend fill:#10b981,stroke:#047857,stroke-width:2px,color:white;
    classDef ml fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:white;
    classDef external fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:white;

    %% Nodes
    User(("👨‍💻 User / Student"))
    
    subgraph Frontend ["Frontend (React.js on Vercel)"]
        UI_Assess["Assessment Page"]
        UI_Chat["AI Companion Chat"]
        UI_Scraper["Social Scrapers"]
    end
    
    subgraph Backend ["Backend API (Flask / Python 3.10)"]
        API_Gateway["API Routing / Controllers"]
        
        subgraph MachineLearning ["PyTorch ML Engine"]
            Predictor["Predictor Module"]
            Model_Weights[("Model Weights (.pt)")]
        end
        
        subgraph DataScraping ["Scraping Engine"]
            Scraper_Reddit["Reddit JSON Fetcher"]
            Scraper_Twitter["Twitter Headless (Selenium)"]
        end
    end
    
    subgraph ExternalSources ["External Platforms"]
        Reddit_API["Reddit Public API"]
        Nitter_Mirrors["Rotating Nitter Mirrors"]
    end

    %% Connections
    User -->|Interacts| Frontend
    
    UI_Assess -->|POST /api/predict| API_Gateway
    UI_Chat -->|POST /api/chat| API_Gateway
    UI_Scraper -->|POST /api/scrape/*| API_Gateway
    
    API_Gateway <-->|Predict Request/Response| Predictor
    Predictor <-->|Loads| Model_Weights
    
    API_Gateway -->|Trigger Scrape| DataScraping
    DataScraping -->|Batch Predict| Predictor
    
    Scraper_Reddit <-->|Fetch JSON| Reddit_API
    Scraper_Twitter <-->|Headless Browsing| Nitter_Mirrors
    
    %% Apply styles
    class User,UI_Assess,UI_Chat,UI_Scraper frontend;
    class API_Gateway,Scraper_Reddit,Scraper_Twitter backend;
    class Predictor,Model_Weights ml;
    class Reddit_API,Nitter_Mirrors external;
```

---

## 2. Core Machine Learning Pipeline (SBERT + BiLSTM)

This is the exact data flow for how a piece of text is processed and classified into an emotional severity level. This is crucial for defending your 87.4% accuracy claim.

```mermaid
flowchart TD
    %% Define styles
    classDef input fill:#f3f4f6,stroke:#9ca3af,stroke-width:2px,color:black;
    classDef process fill:#e0e7ff,stroke:#6366f1,stroke-width:2px,color:black;
    classDef decision fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:black;
    classDef output fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:black;
    classDef critical fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:black;

    A["Raw Text Input"]:::input --> B["Text Cleaning (Remove URLs/Mentions)"]:::process
    
    B --> C["SBERT Encoder (all-MiniLM-L6-v2)"]:::process
    C -->|384-dimensional Semantic Vector| D["BiLSTM Temporal Tracking"]:::process
    D --> E["Attention Mechanism"]:::process
    
    C --> F["Gated Fusion Network"]:::process
    E --> F
    
    F -->|8-Class Logits| G["Base ML Prediction (Levels 0-7)"]:::process
    
    B --> H{"Keyword Safety Net"}:::decision
    
    H -- "No Crisis Words Detected" --> I["Keep Base ML Prediction"]:::process
    H -- "Crisis Words Detected (e.g. 'suicide')" --> J["Force Override to Level 6/7"]:::critical
    
    I --> K["Final Classification Output"]:::output
    J --> K
    
    K --> L["Attach CBT Recommendations"]:::output
```

---

## 3. Social Media Intelligence (Scraper Flow)

How the system analyzes a user's digital footprint across platforms without using paid API keys.

```mermaid
sequenceDiagram
    participant User as React Frontend
    participant API as Flask Backend
    participant Scraper as Scraping Engine
    participant ML as Batch ML Pipeline
    participant External as Reddit/Nitter

    User->>API: POST /api/scrape/twitter {username}
    API->>Scraper: Initialize Twitter Scraper
    
    Scraper->>External: Connect to Random Nitter Mirror
    alt Mirror is Down/Blocked
        External-->>Scraper: Connection Failed
        Scraper->>External: Rotate to Fallback Mirror (1 of 8)
    end
    
    External-->>Scraper: Return HTML Content
    Scraper->>Scraper: Parse top 15 recent tweets using Selenium/BeautifulSoup
    
    Scraper->>ML: Send list of 15 strings
    Note over ML: Batch Prediction is much faster than running one by one.
    ML-->>Scraper: Return [Level, Confidence] for all 15 posts
    
    Scraper->>API: Format JSON Response (Text, Time, URL, Level)
    API-->>User: Display Analysis Dashboard
```

---

## 4. AI Companion Chatbot Flow

How the chatbot recognizes intent and provides Cognitive Behavioral Therapy (CBT) advice dynamically.

```mermaid
flowchart TD
    %% Define styles
    classDef user fill:#bfdbfe,stroke:#2563eb,stroke-width:2px,color:black;
    classDef ml fill:#e9d5ff,stroke:#9333ea,stroke-width:2px,color:black;
    classDef logic fill:#fde68a,stroke:#d97706,stroke-width:2px,color:black;
    classDef response fill:#bbf7d0,stroke:#16a34a,stroke-width:2px,color:black;

    Msg["User Chat Message"]:::user --> ML["ML Severity Assessment"]:::ml
    
    ML -->|Identifies Emotion Level 0-7| Intent{"Advice Seeking Intent?"}:::logic
    
    Intent -- "No (e.g., 'I feel sad')" --> GenEmpathetic["Generate Empathetic Consolation"]:::logic
    Intent -- "Yes (e.g., 'What should I do?')" --> GenAdvice["Retrieve 5 CBT Action Steps for Level"]:::logic
    
    GenEmpathetic --> SafetyCheck{"Is Level >= 6?"}:::logic
    GenAdvice --> SafetyCheck
    
    SafetyCheck -- "Yes (Crisis)" --> AddHelpline["Append Crisis Helplines (0311-7786264)"]:::logic
    SafetyCheck -- "No" --> Finalize["Finalize Payload"]:::logic
    
    AddHelpline --> Finalize
    
    Finalize --> Output["Send Structured Response to UI"]:::response
```
