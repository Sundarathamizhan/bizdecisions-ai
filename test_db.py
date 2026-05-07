import psycopg2

try:
    conn = psycopg2.connect(
        host="crud-postgres-sundar.postgres.database.azure.com",
        database="postgres",
        user="azureuser@crud-postgres-sundar",
        password="Sundar@4575",
        sslmode="require"
    )
    print("✅ Connected successfully!")

    cur = conn.cursor()
    cur.execute("SELECT version();")
    print(cur.fetchone())

    cur.close()
    conn.close()

except Exception as e:
    print("❌ Connection failed")
    print(e)