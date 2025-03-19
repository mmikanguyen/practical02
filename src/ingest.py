#!/usr/bin/env python3

import ollama
import redis
import numpy as np
import os
import fitz
import argparse
import time
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
