from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os, json, pprint

load_dotenv()

def gan_feedback_loop(
    model_generator_llm,  # LLM instance for Model 1
    model_critic_llm,     # LLM instance for Model 2
    prompt,
    require_user_feedback=False,
    require_user_confirmation=False,
    min_score=8,
    max_attempts=5,
    stagnation_threshold=2
):
    """
    Implements a Generative-Adversarial inspired feedback loop between two LLMs using LangChain.
    
    Parameters:
    - model_generator_llm: The LLM instance for Model 1 (e.g., OpenAI model).
    - model_critic_llm: The LLM instance for Model 2.
    - prompt: The initial prompt to generate content.
    - require_user_feedback: Boolean flag for user feedback between iterations.
    - require_user_confirmation: Boolean flag for user confirmation between iterations.
    - min_score: The minimum score to reach before stopping.
    - max_attempts: Maximum number of iterations.
    - stagnation_threshold: Number of iterations with no improvement before stopping.
    
    Returns:
    - A dictionary containing the final results, including the reason for stopping, final score, 
      generated content, critique history, and any user feedback incorporated.
    """
    # Define PromptTemplates
    # Model 1 PromptTemplate
    prompt_optimization_system_prompt = "You are an expert content editor with a mission to enhance the quality, clarity, coherence, engagement, and overall impact of the content provided."
    prompt_optimization_job = """
Your role is to significantly improve the quality of the content below. Focus on enhancing clarity, engagement, logical flow, factual accuracy, and persuasiveness. Make the content more compelling, well-structured, and easy to understand for the intended audience. Incorporate any provided user feedback to ensure alignment with expectations.

Original Content:
{original_content}

Critique:
{critique}

Follow-up Suggestions:
{followup_suggestions}

User Feedback:
{user_feedback}

Provide the improved content below:
"""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_optimization_system_prompt),
            ("user", prompt_optimization_job)
        ]
    )

    model1_chain = prompt_template | model_generator_llm
    
    # Model 2 PromptTemplate
    model2_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert critic specializing in content quality assessment. Your goal is to provide a comprehensive, actionable, and constructive critique of the content below. Your critique should help make the content clearer, more engaging, and more effective in achieving its intended purpose. Provide your response in a structured JSON format."),
            ("user", "{content_to_critique}")
        ]
    )
    model2_chain = model2_prompt_template | model_critic_llm

    # Initialization
    critique_history = []
    user_feedback_incorporated = []
    reason_to_stop = ""
    final_score = 0
    attempts = 0
    stagnation_counter = 0
    previous_score = 0
    user_feedback = ""
    
    # Main loop
    while attempts < max_attempts:
        attempts += 1
        
        # Step 1: Model 1 generates content
        if attempts == 1:
            # First iteration uses the initial prompt
            generated_content = model_generator_llm.invoke(prompt).content
        else:
            # Subsequent iterations use the updated content
            model1_inputs = {
                "original_content": generated_content,
                "critique": critique_json.get('Critique', ''),
                "followup_suggestions": "\n".join(critique_json.get('FollowUpSuggestions', [])),
                "user_feedback": user_feedback
            }
            generated_content = model1_chain.invoke(model1_inputs).content
        
        if require_user_confirmation:
            print("\nGenerated Content:")
            print(generated_content)
            user_input = input("\nPress Enter to continue or type 'stop' to terminate: ")
            if user_input.lower() == 'stop':
                reason_to_stop = "User terminated the process."
                break
        
        # Step 2: Model 2 critiques the content
        model2_inputs = {"content_to_critique": generated_content}
        critique_json = model2_chain.invoke(model2_inputs)

        critique_history.append(critique_json)
        
        final_score = critique_json.get('Score', 0)
        
        # Check for stagnation
        if final_score <= previous_score:
            stagnation_counter += 1
        else:
            stagnation_counter = 0  # Reset if improvement is detected
        previous_score = final_score
        
        # Exit conditions
        if final_score >= min_score:
            reason_to_stop = "Desired score reached."
            break
        if stagnation_counter >= stagnation_threshold:
            reason_to_stop = "No significant improvement detected."
            break
        
        # Step 3: User feedback if required
        if require_user_feedback and critique_json.get('ClarifyingQuestions'):
            print("\nModel 2 has the following questions:")
            for question in critique_json['ClarifyingQuestions']:
                print(f"- {question}")
            user_feedback = input("\nPlease provide answers to the above questions: ")
            user_feedback_incorporated.append(user_feedback)
        else:
            user_feedback = ""
        
        # Optional user confirmation
        if require_user_confirmation:
            print("\nCritique:")
            print(critique_json.get('Critique', ''))
            print(f"Score: {final_score}")
            user_input = input("\nPress Enter to continue or type 'stop' to terminate: ")
            if user_input.lower() == 'stop':
                reason_to_stop = "User terminated the process."
                break
        
    else:
        reason_to_stop = "Maximum attempts reached."
    
    # Compile the final result
    result = {
        "ReasonToStop": reason_to_stop,
        "FinalScore": final_score,
        "ContentGenerated": generated_content,
        "CritiqueHistory": critique_history,
        "UserFeedbackIncorporated": user_feedback_incorporated if user_feedback_incorporated else None
    }
    
    # Print the JSON result in a pretty format
    pprint.pprint(result)
    
    return result


# Load API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Replace 'your-openai-api-key' with your actual API key
generator_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

# Define JSON schema for critic LLM
json_schema = {
    "title": "CriticFeedback",
    "description": "Comprehensive feedback provided by the critic model to improve the content quality effectively.",
    "type": "object",
    "properties": {
        "Critique": {
            "type": "string",
            "description": "A detailed critique of the content, focusing on clarity, engagement, accuracy, and persuasiveness. Include specific examples of what can be improved and why."
        },
        "ClarifyingQuestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific questions aimed at clarifying ambiguities or obtaining additional information that could help improve the content further."
        },
        "Score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "A score between 0 and 100 to rate the content's quality, where higher scores indicate better overall quality and alignment with goals."
        },
        "FollowUpSuggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Actionable suggestions for enhancing specific aspects of the content, such as structure, style, tone, factual accuracy, or engagement strategies."
        }
    },
    "required": ["Critique", "ClarifyingQuestions", "Score", "FollowUpSuggestions"]
}

critic_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key).with_structured_output(json_schema)

input_prompt = input("Add here your request: ")

result = gan_feedback_loop(
    model_generator_llm=generator_llm,
    model_critic_llm=critic_llm,
    prompt=input_prompt,
    require_user_feedback=False,
    require_user_confirmation=False,
    min_score=85,
    max_attempts=6,
    stagnation_threshold=3
)


print("Reason to Stop:", result['ReasonToStop'])
print("Final Score:", result['FinalScore'])
print("Generated Content:\n", result['ContentGenerated'])
