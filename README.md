# Emotional Memory Agent

A multi-tenant emotional memory system powered by an LLM agent, featuring per-user privacy isolation, long-term memory persistence, RAG-based retrieval, diary writing, and QQ private/group chat support.

## Overview

Emotional Memory Agent is a personalized companionship chatbot project designed for QQ conversations.  
It combines an LLM agent, retrieval-augmented generation (RAG), long-term memory, and tool calling to build a more continuous and emotionally aware chat experience.

The system supports:

- personalized conversation based on user-specific memory
- diary writing as a callable tool
- persistent chat history across restarts
- retrieval from local text and PDF documents
- multi-user isolation based on QQ ID
- QQ private chat and group chat interaction through WebSocket

## Features

- **Multi-tenant memory isolation**  
  Each user's memory is tagged with their QQ ID, and retrieval is filtered by the current user identity.

- **Long-term memory**  
  Diary entries and chat history are stored locally, allowing the system to preserve memory across sessions and restarts.

- **RAG-based retrieval**  
  The agent retrieves relevant content from local diaries and uploaded reference documents before generating responses.

- **Tool calling for diary writing**  
  When the user asks to save memories or write a diary, the agent can call a dedicated tool to store the content.

- **Persistent message history**  
  Chat history is stored as JSON files using file-based message history.

- **QQ private and group chat support**  
  The bot can respond in private chats and selectively react in group chats when triggered.

## Architecture

The project mainly consists of the following modules:

1. **LLM Brain**  
   Responsible for dialogue generation, tool decision-making, and reasoning.

2. **Vector Memory Store**  
   Stores diary/document embeddings and supports semantic retrieval with metadata filtering.

3. **Diary Writing Tool**  
   Saves user memories into local files and synchronizes them into the vector database.

4. **Persistent Chat History**  
   Stores per-user conversation history in JSON files for long-term continuity.

5. **QQ Messaging Interface**  
   Connects to QQ through WebSocket and supports both private and group chat workflows.

## Privacy Design

This project uses **QQ ID as the user identity key**.

To avoid cross-user data leakage:

- each memory document is tagged with `user_qq`
- retrieval is performed with metadata filtering
- chat history is stored separately for each user

This design helps ensure that one user's memories are not retrieved for another user.

## Project Structure

```text
.
├── main.py
├── diaries/
│   └── <user_qq>/
├── chat_history/
│   └── <user_qq>.json
├── requirements.txt
├── .env.example
└── README.md
