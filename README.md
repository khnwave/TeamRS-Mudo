# 무한도전 짤 추천 시스템 : 짤로만 소통하는 커뮤니티

### 🌐 프로젝트 개요

본 서비스는 국민대학교 D&A(빅데이터 분석 학회), X:AI(인공지능 학회), Wink(웹 학술동아리)가 공동 주최한  
‘제 2회 D&X:W Conference’ 에 출품된 AI Tech Track – Team RS의 서비스입니다.

‘무한도전 짤 추천 시스템 : 짤로만 소통하는 커뮤니티’는  
사용자가 입력창에 글을 작성하면 문맥과 사용자 행동 로그를 기반으로  
가장 적합한 무한도전 짤(Meme)을 자동 추천하는 서비스입니다.  

이를 통해 텍스트 없이 짤로만 대화하는 새로운 형태의 커뮤니티를 구현하고자 했습니다.

### ⚙️ 주요 기능

1. 맥락 기반 추천 (Semantic-level Recommendation)  
   : 사용자의 감정이나 상황을 분석하여, 그 맥락에 가장 적합한 무한도전 짤을 자동으로 추천합니다.

2. 개인화 추천 (Personalized Ranking) <br>
   : 사용자 로그 데이터를 기반으로 개인별 선호도를 학습하여, 사용자가 더 좋아할 만한 짤을 우선적으로 노출합니다.

### 🧠 기술적 특징
Sentence Transformer를 활용하여 문맥을 이해하며, LightFM을 통해 개인의 성향을 반영함으로써,  
정확도와 개인화를 모두 달성한 짤 추천 시스템을 구현했습니다.

### 👥 팀 멤버
김정호, 김해나, 문유하, 송민승

<br>

# Mudo Meme Recommendation System : The Meme-Only Conversation Platform

### 🌐 Overview

This service was developed by Team RS and presented at the AI Tech Track of the  
2nd D&X:W Conference, jointly organized by D&A (Big Data Analysis Society),  
X:AI (Artificial Intelligence Society), and Wink (Web Development Society) of Kookmin University.

The “Mudo Meme Recommendation System: The Meme-Only Conversation Platform”  
automatically recommends the most contextually appropriate Infinite Challenge memes  
when a user types a message in the input field.  

Through this system, we aim to create a new form of community  
where communication takes place solely through memes instead of text.

### ⚙️ Core Features

1. Context-based Recommendation (Semantic-level Recommendation)  
   : Analyzes the user’s emotions and context to automatically recommend Infinite Challenge memes that best match the situation.

2. Personalized Ranking  
   : Learns from user log data to model individual preferences, prioritizing memes that each user is more likely to prefer.

### 🧠 Technical Highlights
By utilizing Sentence Transformers for contextual understanding  
and LightFM for personalized modeling,  
we built a meme recommendation system that achieves both accuracy and personalization.

### 👥 Team Members
Jeongho Kim, Hannah Kim, Yooha Moon, Minseung Song
