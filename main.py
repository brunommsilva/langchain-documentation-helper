#!/usr/bin/env python3

import sys

from dotenv import load_dotenv

from classes.agent import Jarvis
from classes.logger import log_info

load_dotenv()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a question as a command line argument.")
        sys.exit(1)

    question = sys.argv[1]

    agent = Jarvis()

    response = agent.answer_question(question)

    log_info(f"Response from Jarvis: {response}")
