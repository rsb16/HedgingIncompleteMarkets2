curl -X GET "https://api.sandbox.amadeus.com/v1.2/flights/extensive-search?apikey=$1&origin=$2&destination=$3&departure_date=$4&one-way=true&direct=true" >> flight-data.json
