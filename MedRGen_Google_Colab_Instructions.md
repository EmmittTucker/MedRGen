# MedRGen Google Colab Setup Instructions

## 📋 Overview

This guide provides step-by-step instructions for running the Medical Reasoning Dataset Generator (MedRGen) on Google Colab. MedRGen creates high-quality synthetic medical cases through simulated doctor-patient interactions.

## 🔧 Prerequisites

### Required:
1. **Google Account** - For accessing Google Colab
2. **OpenAI API Key** - With access to GPT-4 and O1-preview models
3. **Internet Connection** - Stable connection for API calls

### Recommended:
1. **Google Colab Pro** - For better performance and longer runtimes
2. **Google Drive** - For external storage of large datasets
3. **Basic Python Knowledge** - For customizing generation parameters

## 🚀 Quick Start Guide

### Step 1: Access the Notebook
1. Open the `MedRGen_Google_Colab.ipynb` notebook in Google Colab
2. Click the "Open in Colab" badge at the top of the notebook
3. Sign in to your Google account if prompted

### Step 2: Runtime Setup
1. Go to **Runtime** → **Change runtime type**
2. Select **Python 3** as the runtime type
3. Choose **GPU** if available (optional but recommended for faster processing)
4. Click **Save**

### Step 3: Execute Setup Cells
Run the first few cells in order to:
1. Check system information
2. Mount Google Drive (recommended for storage)
3. Clone the MedRGen repository
4. Install required dependencies

### Step 4: Configure API Key
1. When prompted, enter your OpenAI API key securely
2. The key will be stored temporarily in the session
3. **Never share your API key publicly**

### Step 5: Generate Medical Cases
1. Start with a single test case to verify setup
2. Scale up to batch generation as needed
3. Monitor API usage and costs

## 📊 Usage Patterns

### Single Case Generation
```python
# Generate one cardiology case
test_case = await generator.generate_single_case(
    specialty="cardiology",
    complexity="moderate",
    symptom_theme="cardiovascular"
)
```

### Batch Generation
```python
# Generate 5 cases for testing
batch_cases = await generate_batch_cases(count=5)

# Generate 25 cases for development
batch_cases = await generate_batch_cases(count=25)

# Generate 100+ cases for production (requires Colab Pro)
batch_cases = await generate_batch_cases(count=100)
```

### Configuration Options
- **Specialties**: cardiology, internal_medicine, emergency_medicine, family_medicine
- **Complexities**: simple, moderate, complex
- **Themes**: cardiovascular, respiratory, gastrointestinal, neurological

## 💰 Cost Management

### API Usage Estimates
- **Single Case**: ~$0.50-$2.00 (depending on complexity)
- **Batch of 25**: ~$12-$50
- **Batch of 100**: ~$50-$200

### Cost Optimization Tips
1. Start with small batches for testing
2. Use "simple" complexity for initial testing
3. Monitor OpenAI dashboard for usage
4. Consider using GPT-4-turbo instead of O1-preview for cost savings

## 🔧 Troubleshooting

### Common Issues

#### API Key Errors
```
Error: Incorrect API key provided
```
**Solution**: Verify your OpenAI API key and model access permissions

#### Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce batch size
- Restart runtime
- Upgrade to Colab Pro

#### Import Errors
```
ModuleNotFoundError: No module named 'main'
```
**Solution**:
- Ensure repository was cloned successfully
- Run the Python path setup cell again
- Restart runtime and re-run setup cells

#### Network Timeouts
```
ReadTimeout: The read operation timed out
```
**Solution**:
- Check internet connection
- Retry the operation
- Reduce batch size to avoid long-running requests

### Performance Issues

#### Slow Generation
- **Cause**: API rate limits or network latency
- **Solution**: Add delays between requests, use smaller batches

#### Session Timeouts
- **Cause**: Colab free tier limitations
- **Solution**: Upgrade to Colab Pro or save progress frequently

## 📁 File Management

### Directory Structure
```
/content/drive/MyDrive/MedRGen_Projects/MedRGen/
├── data/
│   └── output/
│       ├── raw_dataset/          # Original generated cases
│       └── edited_dataset/       # Enhanced cases
├── logs/                         # Generation logs
├── src/                          # Source code
└── .env                         # Configuration file
```

### Data Export
1. **Individual Files**: Download from file browser
2. **ZIP Archive**: Automatically created with all datasets
3. **CSV Summary**: For data analysis in other tools

## 🎯 Best Practices

### For Testing
1. Start with `small_batch` (5 cases)
2. Use "moderate" complexity
3. Test one specialty first
4. Verify output quality before scaling

### For Development
1. Use `medium_batch` (25 cases)
2. Mix complexities and specialties
3. Enable quality evaluation
4. Export in multiple formats

### For Production
1. Use `large_batch` (100+ cases)
2. Focus on specific specialties
3. Always run enhancement pipeline
4. Validate with medical professionals

## 🔒 Security Considerations

### API Key Safety
- Never commit API keys to version control
- Use Colab's secure input for API keys
- Rotate keys regularly
- Monitor usage for unauthorized access

### Data Privacy
- Generated cases are synthetic (not real patient data)
- Review cases before sharing publicly
- Follow your organization's data policies
- Consider HIPAA compliance if applicable

## 📞 Support

### Getting Help
1. **Documentation**: Check the main README.md
2. **Issues**: Create GitHub issues for bugs
3. **Community**: Join discussions in the repository
4. **Commercial**: Contact for licensing questions

### Common Resources
- OpenAI API Documentation
- Google Colab Documentation
- Python AsyncIO Guide
- Medical Terminology References

## 🚀 Next Steps

After successful setup:
1. Generate your first dataset
2. Analyze the results
3. Scale to production volumes
4. Integrate with your ML pipeline
5. Consider commercial licensing

---

**Happy generating! 🏥✨**

*For the latest updates and documentation, visit the [MedRGen GitHub repository](https://github.com/your-username/MedRGen).*
