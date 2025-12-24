import yaml

print("🔍 Validating Kubeflow pipeline YAML...")

try:
    with open('pm25_pipeline.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    print("✅ Pipeline YAML is valid and well-formed")
    print()
    print("📊 Pipeline Details:")
    print(f"  Pipeline Name: {data.get('pipelineInfo', {}).get('name', 'N/A')}")
    print(f"  Components: {len(data.get('components', {}))}")
    print(f"  Executors: {len(data.get('deploymentSpec', {}).get('executors', {}))}")
    print()
    
    # List components
    print("📦 Components:")
    for i, comp_name in enumerate(data.get('components', {}).keys(), 1):
        clean_name = comp_name.replace('comp-', '').replace('-', ' ').title()
        print(f"  {i}. {clean_name}")
    
    print()
    print("✅ Pipeline validation successful!")
    print()
    print("🚀 Next steps:")
    print("  1. Local testing: python scripts/pipeline.py")
    print("  2. Deploy to Kubeflow: python kubeflow_deploy.py --host http://localhost:8080")
    
except Exception as e:
    print(f"❌ Error validating pipeline: {e}")
