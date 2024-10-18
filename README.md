# Find My Circular(FMC) 

## **Table of Contents**  
1. [Introduction](#introduction)  
2. [The Problem](#the-problem)
3. [What is the cost of storage structure](#what-is-the-cost-of-this-storage-structure)
4. [The Solution](#the-solution)  
5. [Getting Started](#getting-started)  
7. [Project Resources: Learn more about FMC](#project-resources-learn-more-about-fmc)  
8. [Contributing](#contributing)  
9. [About Us](#about-us)

# Introduction
To create a knowledge graph that maps public information from regulators to enable easy access and querying, thereby reducing operational overhead and facilitating efficient discovery, research, and documentation.

# The Problem
How could public authorities in India publish and store information more organized? This creates considerable inefficiencies in the legal sector, given the cumbersome nature of discovering the latest and most relevant regulations, tracing chronology, and retrieving specific information within these documents. This also has repercussions on other stakeholders outside the legal fraternity, for example - banks, businesses, and individuals who need help tracing the most recent and relevant applicable laws within their sector. 

# What is the cost of this storage structure?
- Time lost in search 
- Possibility of misinformation being propagated 
- Cost of doing business is high
- Sectors working in ambiguity

# The Solution
The problem is tackled in 2 parts: 

1. Giving structure to a vast array of regulations through knowledge graphs: 
The knowledge graphs will capture relationships that, in the case of circulars, involve those repealed, amended, etc., by another.

2. Adding ease of use and relevance through RAGs: FMC can chunk the data to retrieve relevant information limited only to circulars and allows user interaction through natural language.

The user can interact with circulars via a chat interface and see outputs cited, inferred, and synthesized by the LLM layer. The system will be automated and scalable. Optimizations can be further made to improve the latency and storage burdens of creating a Knowledge Graph-based RAG solution.

# Get Started 
1. Clone the repository:
   ```
   git clone https://github.com/PeoplePlusAI/FindMyCircular.git
   ```

2. The source code for the individual prototypes can be found in these folders:  
   - [Langraph Knowledge Graph Generator](./langraphKG)
   - [Self-RAG Chatbot](./selfRAG)  

3. Follow the instructions in the README.md for contributing.

# Project Resources: Learn more about FMC
- [FMC Project Charter](https://docs.google.com/document/d/1BqFkP-UGqeTOClv-3czZY-mAIlulWc3cH3YAZ_YVjBs/edit?usp=sharing)
- [Read our blog on the Project](https://peopleplus.ai/blog/ai-can-you-find-my-circular-(fmyc))
- [Read our Technical Documentation](https://docs.google.com/document/d/1-yK_mDoNXKwQN6uyAueuxXDfvlw9eiKse8eujbI1e5E/edit?usp=sharing)

## **Contributing**
We welcome contributions from the community. To contribute to this Project, please follow the guidelines in [CONTRIBUTING.md](./.github/CONTRIBUTING.md).

Please reach out to **[saket@peopleplus.ai](mailto:meghana@peopleplus.ai)** or **[naiema@peopleplus.ai](mailto:naiema@peopleplus.ai)** or **[meghana@peopleplus.ai](mailto:meghana@peopleplus.ai)** if you are interested in learning more about the Project and taking on any of these parts.

## **About Us**  
People+ai connects do-ers, dreamers, tinkerers, and innovators with ideas & resources to build an ecosystem that can empower a billion people to reach their potential. Learn more about other People+ai initiatives **[here](https://peopleplus.ai/)**.

Made with ♥️ for 🇮🇳 by Team People+ai
