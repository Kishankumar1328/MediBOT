import streamlit as st
import pandas as pd
import numpy as np
import json
import torch
import os
import warnings
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')

class AdvancedMedicalChatbot:
    def __init__(self):
        # Load pre-trained SentenceTransformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.medical_data = None
        self.embeddings = None

    def load_data(self):
        try:
            json_path = r"C:\Users\ADMIN\PycharmProjects\medicare\csvjson.json"

            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            self.medical_data = pd.DataFrame(data)

            # Standardize column names
            self.medical_data.rename(columns={
                "Medicine Name": "medicine",
                "Diseases": "condition",
                "Side_effects": "side_effects",
                "Composition": "composition",
                "Manufacturer": "manufacturer",
                "Excellent Review %": "excellent_review",
                "Average Review %": "average_review",
                "Poor Review %": "poor_review",
            }, inplace=True)

            # Combine all text columns into one for embedding
            self.medical_data['combined_text'] = self.medical_data.apply(
                lambda row: ' '.join(str(cell) for cell in row if pd.notna(cell)),
                axis=1
            )

            self.create_embeddings()

        except Exception as e:
            st.error(f"Error loading JSON data: {str(e)}")

    def create_embeddings(self):
        embedding_file = "embeddings.npy"

        if os.path.exists(embedding_file):
            self.embeddings = torch.tensor(np.load(embedding_file))
        else:
            combined_texts = self.medical_data['combined_text'].tolist()
            self.embeddings = self.model.encode(combined_texts, convert_to_tensor=True)
            np.save(embedding_file, self.embeddings.numpy())  # Save embeddings

    def find_best_matches(self, query, top_n=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        top_results = torch.topk(cosine_scores, k=top_n)

        recommendations = []
        for score, idx in zip(top_results.values, top_results.indices):
            data = self.medical_data.iloc[idx.item()].to_dict()
            recommendations.append({
                "medicine": data['medicine'],
                "condition": data['condition'],
                "similarity_score": float(score.item()),
                "composition": data.get("composition", ""),
                "manufacturer": data.get("manufacturer", ""),
                "side_effects": data.get("side_effects", "")
            })

        return recommendations

    @staticmethod
    def filter_general_fever_recommendations(recommendations):
        general_fever_recommendations = []
        for rec in recommendations:
            condition = rec['condition'].lower()
            if 'fever' in condition and not any(specific in condition for specific in ['dengue', 'typhoid', 'malaria']):
                general_fever_recommendations.append(rec)
        return general_fever_recommendations


@st.cache_resource
def load_chatbot():
    chatbot = AdvancedMedicalChatbot()
    chatbot.load_data()
    return chatbot


def main():
    st.set_page_config(
        page_title="MedBot Assistant with AI",
        page_icon="🧠",
        layout="wide",
    )

    st.title("💊 AI-Powered MedBot Assistant")
    st.markdown("---")

    chatbot = load_chatbot()

    query = st.text_input("Enter your medical question or symptoms")

    if query:
        recommendations = chatbot.find_best_matches(query)

        if query.lower() == 'fever':
            recommendations = chatbot.filter_general_fever_recommendations(recommendations)

        if recommendations:
            st.markdown("### 🩺 Top Medication Recommendations:")
            for rec in recommendations:
                st.markdown(f"""
                    - **Medicine:** {rec['medicine']} (Score: {rec['similarity_score']:.2f})
                    - 🎯 Condition: {rec['condition']}
                    - 💉 Composition: {rec['composition']}
                    - 🏭 Manufacturer: {rec['manufacturer']}
                    - ⚠️ Side Effects: {rec['side_effects']}
                    ---
                """)
        else:
            st.warning("No relevant results found. Please try rephrasing your query.")


if __name__ == "__main__":
    main()
