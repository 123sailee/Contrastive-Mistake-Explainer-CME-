# MedGuard AI - Clinical Decision Support System
# Streamlit application for medical AI failure detection and correction

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import sys
import warnings
import joblib
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import MedGuard components
from model_trainer import ModelTrainer
from cme_explainer import CMEExplainer
from data_loader import load_heart_disease
