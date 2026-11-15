# Paperforge

> Paste any arXiv deep learning paper and AI reads it, then builds its architecture as a 3D model that assembles itself block by block.

## Demo moment

Paste the arXiv link for 'Attention Is All You Need'. A paper page flips, highlights sweep across the method section, and pieces fly out of the text: embeddings, multi-head attention, feed-forward blocks. They click together into a full 3D encoder-decoder, each block labeled with its equation and tensor shape, then a data pulse runs through it.

## MVP features

- Fetches arXiv PDF and extracts the architecture with an LLM
- Turns it into a typed block graph with shapes and hyperparameters
- Self-assembling 3D exploded view, step by step
- Hover any block to see the source sentence from the paper

## Stack

Bun + TypeScript server with pdf.js text extraction, Claude for structured architecture JSON, Three.js front end for assembly.

Tags: LLM, PDF parsing, Three.js

## Why share it

Everyone has a paper they never fully understood; watching it build itself in 3D is the explainer they'll forward to their whole team.

Industry: Deep learning – research papers & model understanding
