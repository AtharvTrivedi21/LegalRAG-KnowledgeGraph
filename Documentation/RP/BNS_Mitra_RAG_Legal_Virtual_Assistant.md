# BNS Mitra: RAG-optimized LLM based AI-powered Legal Virtual Assistant

---

**Bhavesh Patil**  
Amity Institute of Information Technology  
Amity University Rajasthan  
Jaipur, India  
bhaveshpatil1978@gmail.com

**Rizwan Alam**  
Amity Institute of Information Technology  
Amity University Rajasthan  
Jaipur, India  
riz.alig@gmail.com

**Tejas V. Kalal**  
Amity Institute of Information Technology  
Amity University Rajasthan  
Jaipur, India  
tejaskalal68@gmail.com

**Narendra Singh Kushwaha**  
Unitedworld School of Law  
Karnavati University  
Gandhinagar, Gujarat, India  
Narendrasinghkushwaha198@gmail.com

**Pulkit Porwal**  
Amity Institute of Information Technology  
Amity University Rajasthan  
Jaipur, India  
pulkitporwal.dev@gmail.com

**Nimiti Sharma**  
Unitedworld Institute of Technology  
Karnavati University  
Gandhinagar, Gujarat, India  
nimitisharma1605@gmail.com

---

*2025 3rd International Conference on Communication, Security, and Artificial Intelligence (ICCSAI)*  
*Copyright © IEEE–2025 ISBN: 979-8-3315-3607-7*  
*DOI: 10.1109/ICCSAI64074.2025.11063757*

---

## Abstract

This paper presents BNS Mitra, a Virtual legal assistant for Bharatiya Nyaya Sanhita (BNS), which is powered by Artificial intelligence and is designed with RAG (Retrieval Augmented Generation) framework. Since many stakeholders exist: law enforcement, advocates, judiciary members and common people, BNS Mitra assisted by Meta's LLAMA 2 model is helpful in translating user-described incidents into legal sentences to determine the most relevant sections of BNS. BNS Mitra incorporated FAISS (Facebook AI Similarity Search) that provides the encoded BNS database for encoding vectors to use as a database suitable for user queries, thus allowing contextual legal advice to be provided in due time. Due to its simple design, the general public is able to obtain legal information. On the other hand, the time spent carrying out manual research is reduced since professionals are able to quickly obtain relevant sections. In tests BNS Mitra managed to achieve 87% accuracy regarding which BNS sections were most closely related to user input. The software performed well enough in simple cases, but some issues were identified in more complex cases. This paper presents the capacity of AI as a tool to diminish the knowledge gap in the Indian legal system and recommends the idea of expanding the scope of BNS Mitra to other legal areas/fields and language models.

**Keywords:** Legal Virtual Assistant, Bharatiya Nyaya Sanhita (BNS), Retrieval-Augmented Generation (RAG), Meta LLAMA 2, Legal Information Retrieval.

---

## I. Introduction

The application of artificial intelligence (AI) in the legal field has created new avenues for improving the accuracy, efficiency, and accessibility of legal procedures in recent years [1]. The employment of AI-powered Legal Virtual Assistants (LVAs), which provide instant access to pertinent legal information and support legal research, case preparation, and judgment authoring, is one noteworthy advancement in this field [2]. The new RAG-optimized Large Language Model (LLM)-based AI-powered Legal Virtual Assistant "BNS Mitra," designed specifically for the Bharatiya Nyaya Sanhita (BNS), is presented in this study. Law enforcement officials, advocates, judges, and law students are just a few of the stakeholders in the legal and judicial ecosystem for whom BNS Mitra is intended to be a comprehensive tool. In addition to serving as a useful tool for professional application and legal education, BNS Mitra's ability to identify pertinent BNS sections by utilizing cutting-edge AI capabilities helps to close the gap between the judiciary and laypeople.

Objective of this work is to create an easily navigable platform that helps users find the relevant BNS sections for occurrences that have been reported. BNS Mitra democratizes access to legal knowledge by offering the general public a simple interface that makes recommendations for pertinent legal provisions based on reported instances. The intricate terminology and subtleties of legal statutes can be difficult for anyone without specialized legal training to understand. By converting user inquiries into intelligible legal suggestions, BNS Mitra tackles this difficulty, lowering barriers to legal understanding and promoting informed civic participation. By doing this, it hopes to expedite preliminary legal consultations and assist laypeople in making well-informed decisions about possible legal actions or remedies without having to immediately visit legal professionals.

BNS Mitra offers advocates and legal professionals a major benefit by rapidly locating relevant legal passages associated with a particular case, improving research effectiveness, and freeing up practitioners to concentrate on case strategy. By offering focused legal references within the BNS framework, this tool reduces the amount of time practitioners must spend on laborious manual searches, freeing up more time for case-specific preparation and analysis. BNS Mitra is also intended to be a useful tool for law students, providing them with a learning resource to enhance their comprehension of the BNS sections and their uses. BNS Mitra facilitates a more thorough legal education experience by allowing students to link theoretical knowledge with real-world circumstances through the simulation of real-life cases and the recommendation of pertinent BNS sections.

BNS Mitra's usefulness is further demonstrated in the judiciary, where it aids judges and legal officers in confirming the correctness of BNS sections used in charge sheets and serves as a guide for writing judgments. Because legal drafting is so intricate, BNS Mitra's suggestions can help ensure that legal requirements are applied accurately and consistently by fostering a more thorough review process. This approach can also help law enforcement officials, especially when it comes to the registration of FIRs (First Information Reports) and the production of charge sheets, when precise sectioning is essential to maintaining the integrity of the legal system in criminal cases.

BNS Mitra blends the power of language models with real-time legal data retrieval through RAG (Retrieval-Augmented Generation) optimization, guaranteeing that its answers are not only precise but also contextually consistent with the most current BNS updates. This RAG-optimized method increases BNS Mitra's dependability and usefulness as a legal assistant by enabling it to make contextually appropriate recommendations. By developing a complete support system that meets the needs of various users across the legal spectrum, BNS Mitra therefore tackles the intricate demands of India's court system and helps to make the legal system more effective, knowledgeable, and easily accessible.

This study aims to bridge the gap between the judiciary and laypeople by creating an intuitive AI-powered legal virtual assistant that suggests pertinent BNS (Bharatiya Nyaya Sanhita) sections for reported instances. It can also be used as a learning tool by law students to improve their comprehension of BNS sections and help advocates locate pertinent sections for case preparation in a timely manner. The system will also benefit judges by allowing them to verify the accuracy of the applied sections on accused in chargesheet and utilize it as a reference when drafting judgments. The AI-powered Legal Virtual Assistant addresses the needs of various stakeholders in the judiciary and law enforcement agencies as police officials can also utilize AI-powered Legal Virtual Assistant for FIR registration and charge sheet preparation, thereby contributing to a more informed and efficient legal process.

This work aims to answer the following research questions:

- **RQ1.** How accurate are the BNS sections provided by the chatbot?
- **RQ2.** Does prompt engineering improve the legal description of an incident provided by users?
- **RQ3.** How feasible is RAG for information retrieval in the legal domain?

This work proposes a framework for retrieving a Bhartiya Nyaya Sanhita (BNS) section based on a user's legal query. It begins with the user submitting a query, which is then passed to the Meta LLAMA 2 model. The query is rephrased into legal terminology to facilitate easy retrieval of context from the knowledge base that contains a PDF which contains the information about the BNS sections and their description.

Next, the rephrased query is encoded into input vectors. These vectors are compared with document vectors stored in a vector database using FAISS for similarity search. The most relevant document vectors are retrieved from the knowledge base.

The retrieved vectors provide the context required for answering the user's query. This context is passed back to the LLAMA 2 model, which generates the final query. The system then returns the final output to the user, presenting the appropriate BNS sections based on the input query and retrieved context.

To the best of our knowledge, no existing virtual assistant for Bharatiya Nyaya Sanhita (BNS) effectively addresses law-related queries or provides accurate, up-to-date legal information. This work presents the first chatbot designed to retrieve relevant BNS sections based on informal user queries or descriptions of crimes or incidents, offering a novel solution for legal information retrieval within the BNS framework. As the Bhartiya Nyaya Sanhita (BNS) is new to the public, the proposed system can serve as a valuable tool, providing legal assistance and helping users navigate BNS-related queries efficiently.

The remaining sections of the paper are organized as follows: In Section 2, existing literature on AI based tools for legal aid is extensively reviewed. Section 3 outlines the methodology for development of AI-powered Legal Virtual Assistant BNS Mitra. Section 4 presents the results of this study, and at last Section 5 presents the conclusion and future scope of this work.

---

## II. Literature Review

A number of studies emphasize use of Artificial intelligence in legal domain. In a work focusing on AI based legal technology highlights various AI tools in legal practice allowing quicker processes which allows for increased efficiency, but it also raises ethical issues like its accuracy, transparency and bias as well as job displacement [3]. Other researchers also examined how AI might help to provide greater legal aid and better access to criminal justice, by potentially simplifying aspects of legal process like case management, legal research and decision-making [4]. Researchers have explored the synergy between AI and law, namely how AI supports legal reasoning, case-based reasoning, and legal argumentation. Open access preview, which covers applications in legal education, decision making and case analysis, challenges the integration of AI into complex legal frameworks, empowerment of legal process and understanding of jurisprudence [5].

In a research paper titled "DISC-LawLLM: Fine-tuning Large Language Models Based on LaMD for Intelligent Legal Services," the authors designed DISC-LawLLM, which is a model that fine-tuned AI models specifically for the Chinese law domain. Legal syllogism prompted (no pun intended) it to compose the training data, making this model capable of reasoning legally. The model incorporates a retrieval module for accurate and up-to-date information sampling. Hence, shows the effectiveness of DISC on a new benchmark for legal evaluations, called DISC-Law-Eval, including both objective and subjective ones [6].

In another study, a chatbot Bettercall AI-based Legal Assistant presents a tool for making legal information in India clear, especially to the marginalized people. This chatbot employs NLP approaches in further vectorizing the legal text in order to carry out effective semantic searches. The conversational agent is device and language friendly targeting a wider audience towards legal education and primary legal assistance [7].

In a survey Artificial Intelligence in Law the authors consider how to utilize AI for the purposes of performing legal tasks, especially, predicting the outcome of court processes, preparing papers, and organizing voluminous amounts of data. Handling information rich subjects like copyright and patent law, the research illustrates the influence of AI on proficiently accomplishing tasks thus enabling lawyers to concentrate on higher order cognitive work [8].

Xie et al. present DeliLaw, a legal advice service that employs an advanced language model from China with the aim of improving legal information retrieval. The legal retrieval and case retrieval modules coexist within one active system to bolster semantic comprehension and limit hallucination of the model and presents result which is better than university which respond to abstract legal questions [9].

A legal chatbot engineered by Chauhan et al. aims to ensure user assistance with the help of natural language processing and artificial intelligence. The bot engages with users in general law queries, navigates them in carrying out certain legal processes, and provides straightforward answers to frequently asked questions. The key technologies integrated are machine learning (ML) and cosine similarity which alongside the created 2.5GB database which consists of the constitution of India, supreme court decisions, high court rulings, tribunal and commission orders, legal acts and regulations, circulars and notices, legal classification, reports from committees [10].

Kandula et al. created a chatbot to assist lawmens by using NLP and machine learning algorithms. The chatbot was trained on large datasets of legal texts, capable of recognizing key legal terms and returning relevant legal search. The model achieved over 80% accuracy in law retrieval [11].

Nikita et al. proposed LAWBot, which is an AI-driven legal chatbot designed for the Indian legal system. LAWBot utilized Longformer model to generate embeddings for a given document, thereby capturing the semantic context of the text and trained on The dataset obtained from Legal Services Website using web scraping tools and a Lawyer Information Database. Botpress and RASA are utilized for chatbot development. cosine similarity and pattern matching, to help users in legal document retrieval and lawyer searches. The system provided an efficient, user-friendly platform for legal assistance, particularly suited for laymen to navigate complex legal procedures [12].

Mustafa et al. developed an AI-powered chatbot to provide legal advice to small businesses dealing with insolvency. The system leveraged h2oGPT model and zero-shot classification techniques to extract and match from statutory laws and case precedents. Developed bot uses Chroma db for vector storage and semantic search using instructor embeddings [13].

Vakayil et al. created a chatbot using RAG and Llama-2. The model is trained using information from various legal and government resources in India. LangChain is also used as a retriever. developed chatbot achieved 95% accuracy [14].

Amato et al. proposed CREA2 which is a legal conversational agent known designed to help users in resolving legal conflicts within the European Union. This chatbot uses NLP techniques to interpret queries then retrieves relevant legal information using the SBERT model for semantic search. CREA2 helps in dispute resolution services, specifically in inheritance, divorce, and corporate division cases, by utilizing an unsupervised information retrieval system for matching user questions with relevant answers [15].

Surana et al. proposed a chatbot system for crime awareness and complaint registration. NLP is utilized in their study with a custom Named Entity Recognition (NER) model which is trained on a dataset to extract relevant information from user complaints, such as location, crime type, and time of incident. Proposed model also includes a spam detection feature, aiming to minimize cybercrimes through classification models [16].

Firdaus et al. proposed a chatbot to provide accessible legal information, help users to ask questions about applicable laws and receive relevant responses. Proposed chatbot uses Natural Language Processing (NLP), alongwith Levenshtein distance and TF-IDF cosine similarity to interpret non-standard language and find relevant legal documents. The chatbot processes user inputs through various stages, including parsing, lexical analysis, and word correction, ensuring that even incorrectly phrased queries can be accurately understood. A database of legal questions and answers utilized to support the chatbot in delivering precise legal information, helping users better understand laws [17].

Queudot et al. developed chatbots to provide critical legal information based on Canadian government data, focusing on immigration issues, while the second informs bank employees about legal matters relevant to their jobs. The immigration chatbot dataset is publicly available and built from 1088 webpages from Canada's Immigration and Citizenship Help Desk, while another chatbot relies on a smaller dataset from the National Bank of Canada. The use of MLflow ensures reproducibility and efficient model development but both systems face challenges in handling small datasets and unseen vocabulary, particularly in legal conversations [18].

A direct comparison with other AI-powered legal assistants reveals both the advantages and disadvantages of BNS Mitra, even though it is designed for legal issues under Bharatiya Nyaya Sanhita (BNS). To ensure transparency in evaluating its performance, a formal assessment methodology has been integrated to compare retrieval accuracy, scope, and AI-based reasoning skills. Further benchmarking and qualitative evaluations will be part of future improvements to improve its accuracy and effectiveness.

---

## III. Methodology

The proposed framework combines multiple techniques to accelerate user interactions and the processing of legal documents. In order to deliver precise answers, it effectively searches legal documents, translates colloquial queries into formal legal language, and extracts relevant BNS sections. An easy-to-use interface that integrates search and response generation processes answers user enquiries and provides accurate legal information based on the input entered.

This framework is made up of three primary components: Meta LLaMA 2, Retrieval Chain, and Knowledge Base. As depicted in Figure 1, these three components serve as the framework's foundation. These components can be described as follows:

### Figure 1: Proposed Framework for BNS Mitra

```
┌─────────────────────────────────────────────────────────────────┐
│                        BNS MITRA FRAMEWORK                      │
│                                                                  │
│  USER ──► [Query] ──► META LLAMA 2 ──► Rephrased Query          │
│                            │               (Legal Terminology)   │
│                            │                      │             │
│                            ▼                      ▼             │
│                     RETRIEVAL CHAIN         Input Vectors        │
│                            │                      │             │
│                            │              Similarity Search      │
│                            │               using FAISS           │
│                            │                      │             │
│                     KNOWLEDGE BASE ──► Document Vectors          │
│                     (BNS 2023 PDF)     (Pre-encoded)             │
│                            │                      │             │
│                            └──────────────────────┘             │
│                                       │                          │
│                                   Context                        │
│                                       │                          │
│                             META LLAMA 2 (Final)                 │
│                                       │                          │
│                                  Final Output                    │
│                                       │                          │
│                                      USER                        │
└─────────────────────────────────────────────────────────────────┘
```

### A. LLM (Meta LLAMA 2)

The LLM Meta LLaMA 2 serves as the framework's main processing engine. This approach rephrases user queries with applicable legal terms to enhance information retrieval from our knowledge base. Meta LLaMA 2 uses rephrased questions and contextual data to generate actionable legal insights and statutes, bridging the gap between informal and formal legal language.

### B. Retrieval Chain (Search and Retrieval Component)

The retrieval chain serves as the fundamental component upon which the system's capacity to locate and present pertinent legal information is established. This mechanism is in charge of converting user queries into vectors and conducting similarity searches using FAISS (Facebook AI Similarity Search) [19]. The retrieval chain coordinates the interaction of user input with the knowledge base. It converts the rephrased question to vector format and conducts a similarity search across the knowledge base to find the most semantically relevant legal section. This component enables semantic search, allowing the framework to do efficient nearest neighbour lookups in the FAISS Vector Database. It ensures that the most relevant papers are forwarded to Meta LLaMA 2 for response development.

### C. Knowledge Base (Source of Legal Information)

Definitions of various crimes and the corresponding penalties are included in the knowledge base, the official publication of the Bhartiya Nyaya Sanhita (BNS) 2023. For easy retrieval, these descriptions are pre-encoded as vectors. The knowledge base provides the actual and legal basis for the system's operation. It undergoes an encoding process that converts legal documents into a format that allows for rapid and accurate similarity searches when they are queried. The retrieval chain searches the knowledge base for legal references that inform the final output generated by the LLM.

To establish a comprehensive system for identifying pertinent sections of the Bhartiya Nyaya Sanhita (BNS) based on descriptions of incidents provided by users, this framework integrates three crucial components: large language models (LLMs), a knowledge base, and a retrieval chain. This method enhances legal information retrieval through the synthesis of NLP, LLMs, and Retrieval-Augmented Generation. It merges informal language with formal legal terminology, facilitating accurate and context-aware responses, thereby improving the accessibility and understanding of legal content.

The system integrates a variety of critical technologies, including Streamlit for user interaction, LangChain for language model integration, and FAISS for fast similarity search. Responses are processed and generated using OllamaLLM, which has been optimized for legal understanding. The Bhartiya Nyaya Sanhita (BNS) 2023 dataset was preprocessed by extracting text from PDFs and separating it into coherent parts using RecursiveCharacterTextSplitter [function included in langchain].

For retrieval, the system uses FAISS to store embeddings generated by OllamaEmbeddings, which improves document searches. A custom function, `rephrase_query`, turns informal user enquiries into formal legal language. The retrieval chain combines FAISS with OllamaLLM to retrieve precise and legally accurate results.

A Streamlit-based interface facilitates user interactions by utilizing a dual-chain technique to get pertinent legal portions and generate responses.

---

## IV. Results

The results section provides the responses generated by chatbot for diverse incident scenarios which demonstrates the functionality of chatbot in recommending appropriate Bhartiya Nyaya Sanhita (BNS) sections for provided incident scenarios. Screenshots of the outputs generated by chatbot are provided, along with the detailed descriptions, to demonstrate its capability to understand user queries and provide relevant legal information effectively.

### Query No. 1

As shown in Fig. 2, a user inquired about the seriousness of the offense and punishment for the act of a man labeled 'A' that was committed without the latter's consent as per Bharatiya Nyaya Sanhita.

Further, it provides a more detailed answer from a legal perspective. It draws similar conclusions about the legal position and states that as per the 63 BNS, the act committed by Mr. A can be classified as 'Rape' as it involves non-consensual penetration.

> **Fig. 2. Query No. 1** — BNS Mitra chatbot interface showing response to a query about non-consensual acts, identifying the offense under Section 63 BNS as 'Rape' involving non-consensual penetration, with relevant legal provisions and penalties.

### Query No. 2

As shown in Fig. 3, a user inquired about which section of Bharatiya Nyaya Sanhita (BNS) would be invoked if Mr. A directed livestock into Mr. B's land knowing or having the desire that this will damage Mr. B's crops. First, the AI rephrases the query, which helps in understanding what Mr. A did, and why it is a punishable offense, under section 427 of BNS, to drive cattle into the field of Mr. B's crops intentionally and knowingly. This section deals with cases of causing destruction to someone's property with the intention of doing so. Refraining from the legal interpretation, it proposes that also section 326 would be applicable since this section comprises provisions on Mischief by injury, inundation, fire or explosive substance. This section defines behaviors that intend to harm or cause damage by different means such as causing damage to property by animals. In addition, it addresses section 324 which deals with general mischief more specifically. General mischief involves willful acts done with the aim of inflicting loss or damage to another person's property, and stipulates the penalties pegged against such acts.

> **Fig. 3. Query No. 2** — BNS Mitra chatbot interface showing response to a livestock-damage query, identifying applicable sections: Section 427 (driving cattle into another's crops), Section 326 (Mischief by injury/inundation/fire/explosive substance), and Section 324 (General Mischief).

### Query No. 3

As shown in Fig. 3, a user asks assault-related question that concerns Mr. A and Mr. B. Here, Mr. A used a rod made of iron to hit Mr. B where Mr. B ended up with a dislocated shoulder and several broken bones on his face. The researcher probed which code of the Bharatiya Nyay Sanhita (BNS) would suit this action.

BNS Mitra provides a detailed response, suggesting that Section 326 of the BNS might also apply, as it covers "Mischief by injury, inundation, fire, or explosive substance," and it provides a definition of mischief as anything that injures or endamages another person. Here, the assault with an iron rod falls under the domain of mischief as defined in the BNS in Section 326. The assistant also mentions Section 118 entitled – "voluntarily causing grievous hurt by dangerous weapons" which is relevant to the violence inflicted by the weapon.

> **Fig. 4. Query No. 3** — BNS Mitra chatbot interface showing response to an iron rod assault query, identifying Section 326 (Mischief by injury) and Section 118 (Voluntarily causing grievous hurt by dangerous weapons) as applicable BNS sections.

---

### Answer of Research Questions

#### RQ1. How accurate are the BNS sections provided by the chatbot?

The evaluation process used a test set of 500 targets from a wide range of legal queries collected from mock and real case summaries. Each target was posed to the chatbot which returned a recommendation to an appropriate BNS section. These recommendations from the chatbot were next evaluated by lawyers in order to assess how accurately the sections recommended by the chatbot corresponded to section expectations. BNS Mitra is an initial virtual assistant designed to help laypersons. Future enhancements will focus on making the model more precise and comprehensive for better legal assistance.

The chatbot scored an overall rating of **87%**, which means that for 87% of the recommendations made by the chatbot, the suggested BNS section was fully or closely in line with the expectation of the legal expert. This accuracy was marginally higher for such straightforward cases as (e.g., theft, assault) but dropped for complex or vague cases where a plurality of sections or interpretations of the law could be applied. BNS Mitra performed well in simple cases such as theft and assault. BNS Mitra performed poorly in complicated situations where legal interpretation is context-dependent since legal terminology and section applicability differ according to offense-specific contexts.

#### RQ2. Does prompt engineering improve the legal description of an incident provided by users?

Prompt engineering plays a significant role in enhancing the clarity and precision of the legal descriptions provided by users for incident-based queries in the chatbot system. Through iterative prompt engineering, this project aimed to guide users to describe incidents in terms that align closely with legal terminology, improving the relevance and accuracy of BNS section recommendations.

#### RQ3. What is the applicability of RAG for information retrieval in law?

The potential for using Retrieval-Augmented Generation (RAG) for information retrieval in the legal field is highly encouraging yet presents some obstacles. In this particular instance, RAG was used in the augmentation of a chatbot's function on searching for particular sections of Bhartiya Nyaya Sanhita (BNS) corresponding to an event described by the user. This approach allows the combination of document dense retrieval and generative aspects; therefore, the presence of visual relevant history for the legal information rendered by the chatbot is achieved.

RAG implementation raised the standards of information retrieval by integrating general information and legal linguistics, resulting in an **improvement of BNS sections recommendations by 12% over the non-RAG methods**.

While RAG improved the retrieval process, there were instances of partial matches where the retrieved information was related but not exact, especially in complex multi-section cases. This work incorporates transparency through RAG, giving legal provisions BNS references and explaining NLP transformation and retrieval techniques. Future work can include explainability methods (attention visualization, justification) to increase trust.

---

## V. Conclusion

AI-powered applications hold significant potential in the legal domain by streamlining access to legal information and enhancing decision-making processes. This research presents a framework for retrieving relevant Bhartiya Nyaya Sanhita (BNS) sections based on user legal queries. The system processes user inputs through the Meta LLAMA 2 model, rephrasing them into legal terminology to ensure accurate context retrieval from a structured knowledge base. Utilizing FAISS for similarity search, the system compares input vectors with stored legal document vectors, retrieving the most relevant context to generate precise legal recommendations.

The BNS Mitra chatbot developed using this framework, has been rigorously tested across various legal scenarios and has consistently demonstrated high retrieval accuracy, reinforcing its utility for efficient legal information retrieval tool. This framework offers a valuable tool for stakeholders seeking quick and accurate access to legal provisions.

While the current focus remains on enhancing the accuracy of legal retrieval, future developments will aim to incorporate an interactive consultation mechanism to enhance user engagement and provide more personalized legal assistance. Additionally, future research will explore the integration of advance large language models (LLMs) to enhance performance and expanding the knowledge base with more comprehensive legal documents.

---

## References

[1] Bansal, A., & Aggarwal, P., Applications of Artificial Intelligence in the Legal Domain: Challenges and Opportunities. *Journal of Legal Technology*, vol. 14, pp. 45–67, 2022.

[2] Sharma, M., Integrating AI in Legal Systems: The Role of Virtual Assistants in Enhancing Judicial Efficiency. *Indian Journal of Law & Technology*, vol. 8, pp. 12–30, 2023.

[3] Soukupová, J., AI-based legal technology: A critical assessment of the current use of artificial intelligence in legal practice. *Masaryk University Journal of Law and Technology*, vol. 15, pp. 279–300, 2021.

[4] Campbell, R. W. (2023). Artificial intelligence in the courtroom: The delivery of justice in the age of machine learning. *Revista Forumul Judecatorilor*, vol. 15, 2023.

[5] Tu, S.S., Cyphert, A. & Perl, S.J., Artificial intelligence: Legal reasoning, legal research and legal writing, 25. *Minnesota Journal of Law, Science and Technology*, vol. 105, pp. 25, 2024.

[6] Yue, S., Chen, W., Wang, S., Li, B., Shen, C., Liu, S., ... & Wei, Z., Disclawllm: Fine-tuning large language models for intelligent legal services. *arXiv preprint arXiv:2309.11325*, 2023.

[7] Dhore, M., Vimal, A., Agrawal, A., Bajaj, R., & Barde, R., Bettercall: AI based legal assistant. In *2024 5th International Conference on Image Processing and Capsule Networks (ICIPCN)* pp. 248–256, July 2024.

[8] Faisal, D. R., Darari, F., Al Ghifari, M. I., Zamrud, M. Z., O'Vara, M. C., Tobing, B. C. L., & Lee, O., A Hybrid Virtual Assistant for Legal Domain Based on Information Retrieval and Knowledge Graphs. *Jurnal Ilmu Komputer dan Informasi*, vol. 16, pp. 125–140, 2023.

[9] Xie, N., Bai, Y., Gao, H., Xue, Z., Fang, F., Zhao, Q., ... & Yang, M., DeliLaw: A Chinese Legal Counselling System Based on a Large Language Model. In *Proceedings of the 33rd ACM International Conference on Information and Knowledge Management*, pp. 5299–5303, October 2024.

[10] Chauhan, D., Singh, M., Sharma, A., Narang, H., Vats, S. & Sharma, V., Development of a legal chatbot for comprehensive user support *Asia and the Pacific Conference on Innovation in Technology (APCIT)*, MYSORE, India, vol. 2024, pp. 1–4, 2024.

[11] Kandula, A.R., Tadiparthi, M., Yakkala, P., Pasupuleti, S., Pagolu, P. & Chandrika Potharlanka, S.M., Design and implementation of a chatbot for automated legal assistance using natural language processing and machine learning *Annual International Conference on Emerging Research Areas: International Conference on Intelligent Systems (AICERA/ICIS)*, Kanjirapally, India, pp. 1–6, 2023.

[12] Nikita, Srivastav, E., Patel, A., Singh, A., Sharma, R., Rana, D.P. & Mehta, R.G., LAWBOT: A smart user Indian legal chatbot using machine learning framework *IEEE 9th International Conference for Convergence in Technology (I2CT)*, Pune, India, Vol. 2024, pp. 1–7, 2024.

[13] Mustafa, M.S., Abdulfatah, M.B., Abdulkareem, H.A. & Ashir, A.M. Iraqi legal GPT., *21st International Multi-Conference on Systems, Signals & Devices (SSD)*. Erbil, Iraq, 2024, pp. 545–551, 2024.

[14] Vakayil, S., Juliet, D.S., J, A. & Vakayil, S., RAG-based LLM chatbot using Llama-2. *Circuits and Systems (ICDCS), (Coimbatore, India) 7th International Conference on Devices*, Vol. 2024, pp. 1–5, 2024.

[15] Amato, F., Fonisto, M., Giacalone, M. & Sansone, C., An intelligent conversational agent for the legal domain. *Information*, vol. 14, pp. 307, 2023.

[16] Surana, S., Chekkala, J. & Bihani, P., Chatbot based Crime Registration and Crime Awareness System using a custom Named Entity Recognition Model for Extracting Information from Complaints *International Research Journal of Engineering and Technology (IRJET)* Pune, India. Vol. 08, 2021.

[17] Firdaus, V.A.H. et al., *IOP Conference Series: Materials Science and Engineering*, 830, 022089, 2020.

[18] Queudot, M., Charton, É. & Meurs, M.J., Improving access to justice with legal chatbots. *Stats*, vol. 3, pp. 356–375, 2020.

[19] Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P. E., & Jégou, H., The faiss library. *arXiv preprint arXiv:2401.08281*, 2024.

---

*Authorized licensed use limited to: Nirma University Institute of Technology. Downloaded on August 18, 2025 at 03:42:48 UTC from IEEE Xplore. Restrictions apply.*
