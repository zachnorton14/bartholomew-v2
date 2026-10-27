"""Robustness SFT data: teach the model to handle input the curriculum never shows it.

The graded routes all pair a well-formed question with an answer lifted from period
prose. Nothing in them covers what happens when a visitor types gibberish, drops the
question mark, or asks what year it is. Without that coverage the model falls back on
base-corpus behaviour -- book prose has almost no EOS, so it simply keeps going.

This package builds those rows and pushes them to their OWN dataset repo, kept
separate from the graded synth dataset so the curriculum's provenance guarantees
stay clean.
"""
