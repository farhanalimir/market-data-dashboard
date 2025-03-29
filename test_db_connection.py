from sqlalchemy import create_engine, text

# Your Railway PostgreSQL URL
DATABASE_URL = "postgresql://postgres:GNjXfPQlSVjItzMJioiBpPDfEIfRaduD@interchange.proxy.rlwy.net:13046/railway"

try:
    # Create the SQLAlchemy engine
    engine = create_engine(DATABASE_URL)

    # Try to connect and run a simple query
    with engine.connect() as conn:
        result = conn.execute(text("SELECT NOW()"))
        current_time = result.scalar()
        print("✅ Connected successfully!")
        print("📅 Server time is:", current_time)

except Exception as e:
    print("❌ Connection failed.")
    print("Error:", str(e))
