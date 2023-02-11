class ImportData:
    """Import data from a file.

    Attributes:
        url (str): URL of the file to import.
    """ 

    def __init__(self, url):
        """
        Args:
            url (str): URL of the file to import.
        """
        try:
            self.url = url
            self.file = open(self.url)
        except Exception as e:
            print(f'An error occurred: {e}')
            
    def read_lines_txt(self, sep = " +++$+++ "):
        """Read lines from the txt file.

        Returns:
            list: A list of lists, where each inner list contains the columns
                of a line in the file.
        """
        try:
            splitted_file = self.file.read().split('\n')
            list_of_columns = [line.split(sep) for line in splitted_file]
            list_of_columns = list_of_columns[:-1]
            return list_of_columns
        except Exception as e:
            print(f'An error occurred: {e}')
            return None

            