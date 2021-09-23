import numpy as np

class LatexTableWriter:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def decide_type(value, float_after_comma=-1):
        if type(value) == int or type(value) == np.int32:
            return "{:.0f}".format(value)
        elif type(value) == float or type(value) == np.float64 or type(value) == np.float32:
            if float_after_comma < 0:
                return "{:.2f}".format(value)
            else:
                str_return =  "{:." + str(float_after_comma) + "f}"
                return str_return.format(value)
        elif type(value) == str:
            return value
        else:
            return "type {}: not known".format(type(value))

    def write_table(self, caption: str, data: dict, label: str) -> str:
        """
        Converts table into latex table string
        :param label: latex label to \ref
        :param caption: caption of the table
        :param data: dict {column_name: [data], ...}
        :return: str latex table
        """

        n_columns = len(data.keys())

        latex_table_str = "\\begin{table}[htb]\n\t\\begin{center}\n\t\t\\begin{tabular}{%s}\n" % (
                "|l" * n_columns + "|")

        column_names = list(data.keys())
        latex_table_str += "\t\t\t\\hline\n\t\t\t%s\\\\\n\t\t\t\\hline\\hline\n" % (" & ".join(column_names))

        for i in range(len(data[column_names[0]])):
            row = []
            if i == len(data[column_names[0]])-1:
                for j in range(n_columns):
                    row.append(self.decide_type(data[column_names[j]][i], 3))
            else:
                for j in range(n_columns):
                    row.append(self.decide_type(data[column_names[j]][i]))

            latex_table_str += "\t\t\t" + " & ".join(row) + "\\\\\n"
        latex_table_str += "\t\t\t\\hline\n"

        latex_table_str += "\t\t\\end{tabular}\n"
        latex_table_str += "\t\t\\caption{%s}\\label{tab_%s}\n" % (caption, label)
        latex_table_str += "\t\\end{center}\n\\end{table}"

        return latex_table_str

if __name__ == '__main__':
    latex_table_writer = LatexTableWriter()
    print(latex_table_writer.write_table("Test", {"columnA": [1,2,3], "columnB": [1,2,3]}, "test"))
