from helper_functions import generateBiasAnalysisTable
if __name__ == '__main__':
    model_path = input("Please Enter the path of the model or press enter to use the default path::")
    if model_path == "":
        model_path = 'Part_3/saved_models/best_so_far_after_bias1.pth'
    print("The model path is: ", model_path)
    generateBiasAnalysisTable(model_path)