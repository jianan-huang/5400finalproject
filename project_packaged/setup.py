
from setuptools import setup, find_packages

setup(
    name='financial_news_sentiment',
    version='1.0.0',
    description='A project for predicting stock price movement based on financial news sentiment.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'nltk',
        'matplotlib',
        'seaborn',
        'xgboost',
        'yfinance',
        'beautifulsoup4'
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'run_pipeline=financial_news_sentiment.pipeline:run_pipeline'
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License'
    ],
)
