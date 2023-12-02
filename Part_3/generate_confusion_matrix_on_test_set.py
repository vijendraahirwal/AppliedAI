from helper_functions import generate_consufionMatrix_on_test_set
if __name__ == '__main__':
    
    directory_path = input("Please Enter the path of the daataset folder or press enter to use the default path:")
    if directory_path == "":
        directory_path = r'Part_3\updated_dataset_inshape\test'
    print("The directory path is: ", directory_path)
    model_path = input("Please Enter the path of the model or press enter to use the default path::")
    if model_path == "":
        model_path = r'Part_3/saved_models/best_so_far_after_bias1.pth'
    print("The model path is: ", model_path)
    generate_consufionMatrix_on_test_set(model_path,directory_path)