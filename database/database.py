from flask import Flask, jsonify

USERS = [
    { "name": "Michael Scott", "email": "michael.scott@dundermifflin.com", "role": "Regional Manager", "password": "$2b$10$XqK9J4hF2mN8pL5vR7wT3eH6yB9zC1aD4fG8kM2nP6qS9tV3xW5zA" },
    { "name": "Jim Halpert", "email": "jim.halpert@dundermifflin.com", "role": "Sales Representative", "password": "$2b$10$aB3cD5eF7gH9iJ1kL3mN5oP7qR9sT1uV3wX5yZ7aB9cD1eF3gH5iJ" },
    { "name": "Pam Beesly", "email": "pam.beesly@dundermifflin.com", "role": "Receptionist", "password": "$2b$10$bC4dE6fG8hI0jK2lM4nO6pQ8rS0tU2vW4xY6zA8bC0dE2fG4hI6jK" },
    { "name": "Dwight Schrute", "email": "dwight.schrute@dundermifflin.com", "role": "Assistant Regional Manager", "password": "$2b$10$cD5eF7gH9iJ1kL3mN5oP7qR9sT1uV3wX5yZ7aB9cD1eF3gH5iJ7kL" },
    { "name": "Angela Martin", "email": "angela.martin@dundermifflin.com", "role": "Head of Accounting", "password": "$2b$10$dE6fG8hI0jK2lM4nO6pQ8rS0tU2vW4xY6zA8bC0dE2fG4hI6jK8lM" },
    { "name": "Kevin Malone", "email": "kevin.malone@dundermifflin.com", "role": "Accountant", "password": "$2b$10$eF7gH9iJ1kL3mN5oP7qR9sT1uV3wX5yZ7aB9cD1eF3gH5iJ7kL9mN" },
    { "name": "Oscar Martinez", "email": "oscar.martinez@dundermifflin.com", "role": "Accountant", "password": "$2b$10$fG8hI0jK2lM4nO6pQ8rS0tU2vW4xY6zA8bC0dE2fG4hI6jK8lM0nO" },
    { "name": "Stanley Hudson", "email": "stanley.hudson@dundermifflin.com", "role": "Sales Representative", "password": "$2b$10$gH9iJ1kL3mN5oP7qR9sT1uV3wX5yZ7aB9cD1eF3gH5iJ7kL9mN1oP" },
    { "name": "Phyllis Vance", "email": "phyllis.vance@dundermifflin.com", "role": "Sales Representative", "password": "$2b$10$hI0jK2lM4nO6pQ8rS0tU2vW4xY6zA8bC0dE2fG4hI6jK8lM0nO2pQ" },
    { "name": "Andy Bernard", "email": "andy.bernard@dundermifflin.com", "role": "Sales Representative", "password": "$2b$10$iJ1kL3mN5oP7qR9sT1uV3wX5yZ7aB9cD1eF3gH5iJ7kL9mN1oP3qR" },
    { "name": "Kelly Kapoor", "email": "kelly.kapoor@dundermifflin.com", "role": "Customer Service Representative", "password": "$2b$10$jK2lM4nO6pQ8rS0tU2vW4xY6zA8bC0dE2fG4hI6jK8lM0nO2pQ4rS" },
    { "name": "Ryan Howard", "email": "ryan.howard@dundermifflin.com", "role": "Temp", "password": "$2b$10$kL3mN5oP7qR9sT1uV3wX5yZ7aB9cD1eF3gH5iJ7kL9mN1oP3qR5sT" },
    { "name": "Toby Flenderson", "email": "toby.flenderson@dundermifflin.com", "role": "Human Resources", "password": "$2b$10$lM4nO6pQ8rS0tU2vW4xY6zA8bC0dE2fG4hI6jK8lM0nO2pQ4rS6tU" },
    { "name": "Creed Bratton", "email": "creed.bratton@dundermifflin.com", "role": "Quality Assurance", "password": "$2b$10$mN5oP7qR9sT1uV3wX5yZ7aB9cD1eF3gH5iJ7kL9mN1oP3qR5sT7uV" },
    { "name": "Meredith Palmer", "email": "meredith.palmer@dundermifflin.com", "role": "Supplier Relations", "password": "$2b$10$nO6pQ8rS0tU2vW4xY6zA8bC0dE2fG4hI6jK8lM0nO2pQ4rS6tU8vW" },
    { "name": "Darryl Philbin", "email": "darryl.philbin@dundermifflin.com", "role": "Warehouse Foreman", "password": "$2b$10$oP7qR9sT1uV3wX5yZ7aB9cD1eF3gH5iJ7kL9mN1oP3qR5sT7uV9wX" }
]

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify(USERS)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=54707, debug=True)