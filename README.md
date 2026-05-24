# Risks-assessments-LLMs

Folder [**Revision_1**](https://github.com/KIDA-BfR/Risks-assessments-LLMs/tree/main/Revision_1) contains materials referenced in the rebuttal to the reviewers 
- [Log files of the main negotiaion simulation](https://github.com/KIDA-BfR/Risks-assessments-LLMs/tree/main/Revision_1/October_seminar_logs)
- Prompts used for the simulations of the [preliminary discussion of Campylobacter case](https://github.com/KIDA-BfR/Risks-assessments-LLMs/tree/main/Revision_1/October_seminar_paper_prompts/Preliminary_discussion_Campylobacter) and [main discussion of OPM case](https://github.com/KIDA-BfR/Risks-assessments-LLMs/tree/main/Revision_1/October_seminar_paper_prompts/Main_negotiations_OPM) 
- Results of the [anonymous survey](https://github.com/KIDA-BfR/Risks-assessments-LLMs/tree/main/Revision_1/October_seminar_paper_prompts/Main_negotiations_OPM)

# General description of the project 

Project focuses on the implementation of AI-methods, Large Language Models (LLMs) in particular, for assisting negotiation-centred risks assessment.

The repository contains code materials for the paper: **Tackling One Health Risks - How Large Language Models are Leveraged for Risk Negotiation and Consensus-building**.

The paper focuses on steps 1-4 of ngeotiation-centred risks assessment approach proposed by Ehling-Schulz et al., (2024) [Risk negotiation: a framework for One Health risk analysis](https://pubmed.ncbi.nlm.nih.gov/38812798/) and augments the previously developed pipeline from  Abdelnabi et al. (2024) titled [Cooperation, Competition, and Maliciousness: LLM-Stakeholders Interactive Negotiation](https://arxiv.org/abs/2309.17234). 
Parts of the Langchain [cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/two_agent_debate_tools.ipynb) were used for the prelimiary disussion simulation 

`Video_S1_data` folder contains a script in a form of Jupyter Notebook that performs the data analysis of the Bt simulation scenario 

Note: In the current implementation the  `veto` right is directly specified in the prompts of the agents, see **voting rules** section of the used prompts. Thus, it is not required to additionally specify the veto party at the `initial_prompts_file`
