"""
Test script to explore capitalcom-python library API
"""

try:
    from capitalcom import Client
    import inspect

    # Create a client instance (doesn't need to connect to list methods)
    print("Available methods in capitalcom.Client:")
    print("=" * 60)

    # Get all methods
    methods = [m for m in dir(Client) if not m.startswith('_')]

    for method in sorted(methods):
        attr = getattr(Client, method)
        if callable(attr):
            sig = inspect.signature(attr)
            print(f"  {method}{sig}")

    print("\n" + "=" * 60)
    print("Testing connection with demo credentials...")

    # Try to create instance and see what happens
    try:
        client = Client(
            log='demo@demo.com',
            pas='demo',
            api_key='demo'
        )
        print("✓ Client created successfully")

        # Try to see if there's a connect method or if it auto-connects
        print("\nClient attributes:")
        attrs = [a for a in dir(client) if not a.startswith('_')]
        for attr in sorted(attrs):
            print(f"  - {attr}")

    except Exception as e:
        print(f"✗ Client creation failed: {e}")

except ImportError as e:
    print(f"capitalcom library not installed: {e}")
except Exception as e:
    print(f"Error: {e}")
