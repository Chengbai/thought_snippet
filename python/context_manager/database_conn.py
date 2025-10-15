from contextlib import contextmanager


@contextmanager
def database_connection(db_name):
    # Setup: Open connection
    conn = connect_to_database(db_name)
    print("Database connected")

    try:
        yield conn  # Give connection to user
    except Exception as e:
        # Rollback on error
        conn.rollback()
        print(f"Error: {e}, rolling back")
        raise
    else:
        # Commit on success
        conn.commit()
        print("Changes committed")
    finally:
        # Always close connection
        conn.close()
        print("Database connection closed")


# Usage
with database_connection("mydb") as conn:
    conn.execute("INSERT INTO users VALUES (...)")
    # Automatically commits if no exception
    # Automatically rolls back if exception
