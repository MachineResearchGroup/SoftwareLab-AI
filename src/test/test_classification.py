###########################################
#don't move these lines from that position#
import sys                                #
sys.path.append("../")                    #
###########################################
from service.handler import app
import unittest
import json


class ClassificationTest(unittest.TestCase):
    """The ClassificationTest class tests all classification flow"""

    requirement: str = f"The application shall match the color of "
    f"the schema set forth by Department of Homeland Security"

    def setUp(self) -> None:
        self.app = app.test_client()
        app.config["TESTING"] = True
        return super().setUp()

    def test_Type(self) -> None:
        """This method tests if the data received is of type string."""

        response = self.app.post(
            "/", data=self.requirement)
        response = json.loads(response.get_data(as_text=True))
        self.assertEqual(str, type(response.get("category")))
        self.assertEqual(str, type(response.get("description")))

    def test_exception_0(self) -> None:
        """This method does a generic test for exceptions."""
        
        with self.assertRaises(Exception):
            self.app.post("/", data=10)

    def test_exception_1(self) -> None:
        """This method does a generic test for exceptions."""

        with self.assertRaises(Exception):
            self.app.post(1, data=self.requirement)

    def test_json_size(self) -> None:
        """This method tests the size of the received json according to the establised contract."""

        response = self.app.post(
            "/", data=self.requirement)
        response = json.loads(response.get_data(as_text=True))
        size = len(response)
        self.assertEqual(2, size)

    def test_attributes_integrity(self) -> None:
        """This method tests the integrity of the attributes received in json file."""

        response = self.app.post(
            "/", data=self.requirement)
        response = json.loads(response.get_data(as_text=True))
        self.assertTrue("category" in response)
        self.assertTrue("description" in response)
        self.assertFalse("emptyWords" in response)

    def test_text_integrity(self) -> None:
        """This method tests the integrity of the text received in json file."""

        expected = "The application shall do anything in according by something"
        response = self.app.post("/", data=expected)
        response = json.loads(response.get_data(as_text=True))
        self.assertEqual(expected, response.get("description"))
