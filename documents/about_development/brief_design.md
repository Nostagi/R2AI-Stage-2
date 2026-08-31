* Pipeline Description

```
                    ┌──────────────────────────┐
                    │      User Question       │
                    │   Vietnamese Finance QA  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│  1. TABLE KNOWLEDGE BASE CONSTRUCTION                       │
│                                                             │
│  OCR Documents → Table Extraction → Schema → Metadata       │
│                              │                              │
│                              └→ Table Relationships         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  2. TABLE RETRIEVAL                                         │
│                                                             │
│       Question                                              │
│          │                                                  │
│          ▼                                                  │
│      [RECALL] ── broad candidate generation                 │
│          │                                                  │
│          ▼                                                  │
│     Top-K candidate tables                                  │
│          │                                                  │
│          ▼                                                  │
│     [PRECISION] ── evidence-level verification              │
│          │                                                  │
│          ▼                                                  │
│     Relevant / Supporting Tables                            │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  3. LLM ORCHESTRATION / TEXT-TO-PANDAS                      │
│                                                             │
│  Question + Retrieved Tables + Schemas + Relationships      │
│                    │                                        │
│                    ▼                                        │
│             Query Planning                                  │
│                    ↓                                        │
│             Pandas Generation                               │
│                    ↓                                        │
│             Code Validation                                 │
│                    ↓                                        │
│             Execution / Repair                              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  4. ANSWER                                                  │
│                                                             │
│  Numerical Answer + Pandas Code + Source/Table Evidence     │
└─────────────────────────────────────────────────────────────┘
```
