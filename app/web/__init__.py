"""Shared HTML presentation layer for ContractLens's server-rendered pages.

The app deliberately has no build step and no external assets (see
app/main.py's route docstrings) -- every page is a self-contained f-string.
This package factors the *shared* chrome (design tokens, page shell, nav,
badges) out of app/main.py's four HTML routes so restyling happens in one
place instead of four near-duplicate <style> blocks.
"""
