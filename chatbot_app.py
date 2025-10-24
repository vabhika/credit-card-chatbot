# chatbot_app.py

"""
GenAI + RAG Credit Card Recommender Chatbot
--------------------------------------------
End-to-end chatbot using Streamlit for UI with session state handling.
"""

import streamlit as st
import re
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List

# -------------------- Dataset Simulation --------------------
def simulate_credit_card_data(n=200):
    cards = []
    categories = ["Travel", "Shopping", "Cashback", "Premium", "Fuel"]
    for i in range(10):
        cards.append({
            "card_id": i,
            "name": f"Card_{categories[i % len(categories)]}_{i}",
            "category": categories[i % len(categories)],
            "min_income": random.choice([5,8,10,15,20]),
            "annual_fee": random.choice([0,500,1000,2000]),
            "reward_points": random.randint(1000,10000),
            "credit_limit": random.randint(50000,300000),
            "desc": f"This is a {categories[i % len(categories)]} card with benefits on {categories[i % len(categories)]} spends.",
            "best_for": categories[i % len(categories)]
        })
    cards_df = pd.DataFrame(cards)
    users = []
    for _ in range(n):
        income = random.choice([6,10,12,18,25])
        credit_score = random.randint(650,850)
        spend_pref = random.choice(categories)
        card_choice = random.choice(cards_df[cards_df["category"]==spend_pref]["card_id"].values)
        users.append({
            "income": income,
            "credit_score": credit_score,
            "spend_pref": spend_pref,
            "recommended_card": card_choice
        })
    users_df = pd.DataFrame(users)
    return cards_df, users_df

# -------------------- Train Model --------------------
def train_model(users_df):
    le = LabelEncoder()
    users_df["spend_pref_enc"] = le.fit_transform(users_df["spend_pref"])
    X = users_df[["income","credit_score","spend_pref_enc"]]
    y = users_df["recommended_card"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)
    return model, le, list(X.columns)

# -------------------- RAG Setup --------------------
class CardRAGIndex:
    def __init__(self,cards_df):
        self.cards_df = cards_df
        self.embeddings = self._create_embeddings(cards_df["desc"].tolist())
    def _create_embeddings(self,docs):
        import hashlib
        vectors = []
        for text in docs:
            h = hashlib.sha256(text.encode()).hexdigest()
            arr = np.array([int(h[i:i+4],16)%10000 for i in range(0,64,4)])
            vectors.append(arr.astype("float32"))
        return np.stack(vectors)
    def retrieve(self,query,top_k=3):
        q_emb = self._create_embeddings([query])[0]
        sims = np.dot(self.embeddings,q_emb)
        idxs = np.argsort(sims)[::-1][:top_k]
        return self.cards_df.iloc[idxs][["name","desc","best_for"]].to_dict(orient="records")

# -------------------- Explanation --------------------
def explain_with_llm(user_query,recommended_cards,context_snippets):
    explanation = f"Based on your inputs ('{user_query}'), here are top card matches:\n"
    for card in recommended_cards:
        explanation += f"- {card['name']}: Great for {card['best_for']} spends, annual fee ₹{card['annual_fee']}\n"
    explanation += "\nContext retrieved:\n"
    for snip in context_snippets:
        explanation += f"  → {snip['name']}: {snip['desc']}\n"
    return explanation

# -------------------- Input Parsing --------------------
def parse_flexible_input(text: str):
    income_match = re.search(r'(\d+)\s*[lL]',text)
    credit_match = re.search(r'(\d{3,4})',text)
    pref_match = re.search(r'(travel|shopping|cashback|premium|fuel)',text,re.I)
    income = int(income_match.group(1)) if income_match else random.choice([8,10,12])
    credit_score = int(credit_match.group(1)) if credit_match else random.choice([700,750])
    spend_pref = pref_match.group(1).capitalize() if pref_match else random.choice(["Travel","Shopping"])
    return {"income":income,"credit_score":credit_score,"spend_pref":spend_pref}

def build_feature_vector_from_parsed(parsed):
    return {"income":parsed["income"],"credit_score":parsed["credit_score"],
            "spend_pref_enc":{"Travel":0,"Shopping":1,"Cashback":2,"Premium":3,"Fuel":4}[parsed["spend_pref"]]}

# -------------------- Recommend --------------------
def recommend_and_explain(user_text:str,model,rag:CardRAGIndex,cards_df:pd.DataFrame,X_columns:List[str]):
    parsed = parse_flexible_input(user_text)
    row = build_feature_vector_from_parsed(parsed)
    X_sample_df = pd.DataFrame([row])[X_columns]
    probs = model.predict_proba(X_sample_df)[0]
    classes = model.classes_
    idxs = np.argsort(probs)[::-1]
    recs = [{"card_id":classes[i],"prob":float(probs[i])} for i in idxs]
    detailed = []
    for r in recs[:6]:
        row_card = cards_df[cards_df['card_id']==r['card_id']].iloc[0].to_dict()
        detailed.append({**r,"name":row_card['name'],"desc":row_card['desc'],
                         "annual_fee":row_card['annual_fee'],"min_income":row_card['min_income'],
                         "best_for":row_card['best_for']})
    top = detailed[0]
    snippets = rag.retrieve(f"{user_text}. Find product features relevant to {top['name']}.",top_k=3)
    explanation = explain_with_llm(user_text,detailed[:3],snippets)
    return explanation

# -------------------- Streamlit UI --------------------
st.title("GenAI + RAG Credit Card Chatbot 💳")

# Initialize model and RAG once and store in session_state
if 'init_done' not in st.session_state:
    cards_df, users_df = simulate_credit_card_data()
    model, le, X_columns = train_model(users_df)
    rag = CardRAGIndex(cards_df)
    # Store in session_state
    st.session_state.cards_df = cards_df
    st.session_state.model = model
    st.session_state.rag = rag
    st.session_state.X_columns = X_columns
    st.session_state.init_done = True

user_input = st.text_input("You: ", placeholder="Enter your income, credit score, and preference...")

if user_input:
    response = recommend_and_explain(
        user_input,
        st.session_state.model,
        st.session_state.rag,
        st.session_state.cards_df,
        st.session_state.X_columns
    )
    st.text_area("Bot:", value=response, height=300)
