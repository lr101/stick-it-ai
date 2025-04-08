import os
import csv
import psycopg2
from minio import Minio
from PIL import Image, ImageDraw, ImageFont
import io

# Database connection parameters
DB_HOST = '127.0.0.1:5432'
DB_NAME = 'stickprod'
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'

# MinIO connection parameters
MINIO_URL = 'localhost:9000'
MINIO_ACCESS_KEY = 'L8GCU3UXGrVdCBYACwTF'
MINIO_SECRET_KEY = 'B901pfCvAmQwSR90jmyh5qiAHlxdWsw2zXL4mJnX'

# Output CSV file
OUTPUT_CSV = 'annotated_data.csv'

# Connect to PostgreSQL database
def connect_db():
    conn = psycopg2.connect("dbname=stickitprod user=postgres password=postgres")
    return conn

# Fetch important data from the database
def fetch_data(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT p.id, p.latitude, p.longitude, p.description, p.group_id, p.creator_id, g.description, g.name
            FROM pins p
            JOIN groups g ON p.group_id = g.id
            WHERE p.is_deleted = false
        """)
        return cursor.fetchall()

# Download image from MinIO
def download_image(minio_client, pin_uuid):
    try:
        image_data = minio_client.get_object('stick-it-prod', f'pins/{pin_uuid}.png')
        with open(f'annotated_images/pins/{pin_uuid}.png', 'wb') as f:
            f.write(image_data.read())
        return f'annotated_images/pins/{pin_uuid}.png'
    except Exception as e:
        print(f"Error downloading image for {pin_uuid}: {e}")
        return None
def download_group_image(minio_client, group_id):
    try:
        image_data = minio_client.get_object('stick-it-prod', f'groups/{group_id}/group_profile.png')
        with open(f'annotated_images/groups/{group_id}.png', 'wb') as f:
            f.write(image_data.read())
        return f'annotated_images/groups/{group_id}.png'
    except Exception as e:
        print(f"Error downloading image for {group_id}: {e}")
        return None

# Save annotated image


# Save data to CSV
def save_to_csv(data):
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['image_path', 'group_path', 'description', 'group_id', 'user_id', 'group_description', 'group_name'])  # Header
        writer.writerows(data)

# Main function
def main():
    # Connect to the database
    conn = connect_db()
    minio_client = Minio(MINIO_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)

    # Fetch data
    pins_data = fetch_data(conn)
    annotated_data = []

    for pin_uuid, latitude, longitude, description, group_id, user_id, group_description, group_name in pins_data:
        print(f"Processing pin {pin_uuid}")
        # Download image
        image_path = download_image(minio_client, pin_uuid)
        group_image_path = download_group_image(minio_client, group_id)
        if image_path:
            # Append to data for CSV
            annotated_data.append((image_path, group_image_path, description, group_id, user_id, group_description, group_name))

    # Save annotated data to CSV
    save_to_csv(annotated_data)

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main()