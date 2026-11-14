# Research questions (draft)
Written By: Prayash

## Question

Can an AI reader catch the mistakes in a SNAP application, and does that change when an AI agent filled it out?

## Why it matters

Starting in October 2027, the 2025 federal budget law (H.R. 1) requires states with a SNAP payment error rate at or above 6% to pay a share of benefit costs—starting at 5% of total SNAP benefit spending and reaching hundreds of millions or billions of dollars a year for some states (Chohlas-Wood, 2026). In FY2024, 44 states exceeded 6% and the national payment error rate was 10.93%, yet agencies lack the staff to double-check every case manually. Simulations estimate that risk-based review alone costs about seven times as much as combining AI flags with risk scores (Chohlas-Wood, 2026). Faced with steep penalties and tight budgets, agencies have a strong incentive to ship AI-assisted review before they know how accurately an AI reader catches errors.

At the same time, applicants are starting to file with help from AI agents, and agencies are starting to screen with AI readers. Whether errors get caught then depends on how accurate the reader is on AI-filed applications. Prior work has not measured that.

## RQ1. How accurate is an AI reader, by filer?

How well does an LLM reader flag errors in human-filed versus AI-filed applications built from the same real SNAP Quality Control households?

## RQ2. Where do existing review rules fall short?

When reviewers can check only some cases, how much error do existing rules for picking those cases catch as more applications are AI-filed?

## RQ3. Can a better review rule close the gap?

Does a rule that accounts for who filed the application catch more error than the existing rules?
