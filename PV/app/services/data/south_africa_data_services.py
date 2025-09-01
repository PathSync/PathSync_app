# app/services/data/south_africa_data_service.py
import re


class SouthAfricaDataValidator:
    """Validate South African ID numbers and extract information"""

    @staticmethod
    def validate_sa_id_number(id_number):
        """
        Validate South African ID number according to official algorithm
        """
        # Check length
        if len(id_number) != 13 or not id_number.isdigit():
            return False, "ID number must be 13 digits"

        # Check date validity (first 6 digits: YYMMDD)
        year = int(id_number[0:2])
        month = int(id_number[2:4])
        day = int(id_number[4:6])

        if month < 1 or month > 12:
            return False, "Invalid month in ID number"

        if day < 1 or day > 31:
            return False, "Invalid day in ID number"

        # Luhn algorithm check digit validation
        total = 0
        for i in range(0, 12):
            digit = int(id_number[i])
            if i % 2 == 0:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit

        check_digit = (10 - (total % 10)) % 10
        if check_digit != int(id_number[12]):
            return False, "Invalid check digit in ID number"

        return True, "Valid SA ID number"

    @staticmethod
    def extract_info_from_id(id_number):

        # Extract birth date (first 6 digits: YYMMDD)
        year = int(id_number[0:2])
        month = int(id_number[2:4])
        day = int(id_number[4:6])

        # Determine century (currently assumes 1900s for demo)
        birth_year = 1900 + year if year > 22 else 2000 + year

        # Extract gender (7th to 10th digits, gender is determined by the value)
        gender_digit = int(id_number[6:10])
        gender = "Female" if gender_digit < 5000 else "Male"

        return {
            "birth_date": f"{day:02d}/{month:02d}/{birth_year}",
            "age": 2023 - birth_year,  # Simplified age calculation
            "gender": gender
        }