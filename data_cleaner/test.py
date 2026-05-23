from datainspector import DataInspector

inspector = DataInspector()

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

inspector.upload_data(url)

inspector.clean_garbage_values()

inspector.auto_convert_types()

inspector.data_summary()