import random
import json
import csv
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

def generate_user_data():
    # Simulate random timestamps for events
    signup_date = fake.date_this_decade(before_today=True, after_today=False)
    last_login_date = signup_date + timedelta(days=random.randint(0, 365))
    upgrade_date = signup_date + timedelta(days=random.randint(30, 180)) if random.random() > 0.7 else None
    cancellation_date = signup_date + timedelta(days=random.randint(30, 365)) if random.random() > 0.5 else None

    return {
        "user_id": fake.uuid4(),
        "name": fake.name(),
        "email": fake.email(),
        "geographic_location": fake.country(),
        "signup_date": signup_date.isoformat(),
        "last_login_date": last_login_date.isoformat(),
        "subscription_tier": random.choice(["basic", "pro", "enterprise"]),
        "missed_payments": random.randint(0, 3),
        "canceled_subscription": cancellation_date.isoformat() if cancellation_date else None,
        "upgraded_subscription": bool(upgrade_date),
        "upgrade_date": upgrade_date.isoformat() if upgrade_date else None,
        "customer_support_tickets": random.randint(0, 10),
        "ticket_resolved": random.choice([True, False]) if random.random() > 0.7 else None,
        "churned": cancellation_date is not None
    }

# Generate 1,000 synthetic user data entries
synthetic_users = [generate_user_data() for _ in range(1000)]

# Save data to JSON
with open('synthetic_saas_users.json', 'w') as json_file:
    json.dump(synthetic_users, json_file, indent=4)

# Save data to CSV
csv_columns = synthetic_users[0].keys()
with open('synthetic_saas_users.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(synthetic_users)

print("Synthetic SaaS user data saved as 'synthetic_saas_users.json' and 'synthetic_saas_users.csv'.")
