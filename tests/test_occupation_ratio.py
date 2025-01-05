import unittest
from occupation_ratio import calculate_occupation_area_shapely

class TestCalculateOccupationAreaShapely(unittest.TestCase):

    def test_single_box(self):
        preds = {
            "predictions": [{
                "x": 500.0,
                "y": 500.0,
                "width": 1000,
                "height": 1000,
                "class": "hand",
                "confidence": 0.943
            }]
        }
        roi = 1000 * 1000
        result = calculate_occupation_area_shapely(preds, roi)
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_multiple_boxes(self):
        preds = {
            "predictions": [{
                "x": 500.0,
                "y": 500.0,
                "width": 1000,
                "height": 1000,
                "class": "hand",
                "confidence": 0.943
            }, {
                "x": 504.5,
                "y": 363.0,
                "width": 215,
                "height": 172,
                "class": "hand",
                "confidence": 0.917
            }, {
                "x": 400,
                "y": 400,
                "width": 50,
                "height": 52,
                "class": "hand",
                "confidence": 0.87
            }, {
                "x": 78.5,
                "y": 700.0,
                "width": 139,
                "height": 34,
                "class": "hand",
                "confidence": 0.404
            }]
        }
        roi = 1000 * 1000
        result = calculate_occupation_area_shapely(preds, roi)
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_no_boxes(self):
        preds = {
            "predictions": []
        }
        roi = 1000 * 1000
        result = calculate_occupation_area_shapely(preds, roi)
        self.assertAlmostEqual(result, 0.0, places=4)

if __name__ == '__main__':
    unittest.main()