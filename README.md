# 🏠 Calgary Housing Estimator (2024 Assessment Model)

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange?style=for-the-badge)

A machine learning project and interactive web application that predicts 2024 property assessments in Calgary using public city data. The project features advanced data cleaning techniques (Contextual Imputation) and an interactive dashboard for real-time value estimation.

**[🌐 View Live Demo](https://clencytabe.com/projects/calgary-housing)**

---

## 🚀 Features

### 1. Interactive Estimation Dashboard
- Users can input key property features (Quadrant, Type, Land Size, Year Built).
- Real-time price calculation based on the trained regression model logic.
- Visual breakdown of "Location Factor" and "Market Trend" influence.

### 2. Advanced Data Pipeline
- **Contextual Imputation:** Instead of dropping rows with missing `YEAR_OF_CONSTRUCTION`, the system imputes values based on the median year of the surrounding community.
- **Outlier Handling:** Log-transformation (`np.log1p`) applied to assessed values to normalize the skewed distribution.
- **Feature Engineering:**
    - Extracted Quadrants (`NW`, `SW`, `NE`, `SE`) from address strings.
    - Binned Construction Years into "Eras" (e.g., Heritage vs. Modern).
    - Grouped granular Land Use codes (R-C1, R-C2) into simplified categories.

---

## 📊 Data Science Process

### 1. Exploratory Data Analysis (EDA)
We analyzed over **400,000+ records** from the City of Calgary Open Data portal.
- **Discovery:** "Community Code" was the strongest predictor of value after square footage.
- **Discovery:** Older homes (>60 years) in city centers hold value similarly to new builds, while mid-century homes (1970-1990) show the highest depreciation.
- **Irrelevant Features:** House number parity (Odd/Even) showed zero correlation with value and was dropped.

### 2. The Model
The estimation logic is derived from a **Decision Tree Regressor** trained on the cleaned dataset.
- **Target Variable:** 2024 Assessed Value (CAD).
- **Key Predictors:** Land Size (SqM), Community, Building Type, Year of Construction.
- **Performance:** Achieved a Mean Absolute Error (MAE) of **~$51,000 CAD**.

---

## 🛠️ Tech Stack

### Frontend (Interactive Dashboard)
- **Framework:** React + Vite
- **Styling:** Tailwind CSS + Shadcn UI
- **Icons:** Lucide React
- **Hosting:** Vercel

### Data Analysis (The Logic Source)
- **Language:** Python (Jupyter Notebooks)
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Techniques:** Linear Regression, Decision Trees, Random Forest

---

## 📸 Screenshots

| Estimator Dashboard | Data Insights |
|:---:|:---:|
| view at https://clencytabe.com/projects/calgary-housing |

---

## 💻 Running Locally

### Prerequisites
- Node.js (v16+)
- npm or yarn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/calgary-housing-estimator.git
