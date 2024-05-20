
import emoji


def calc_and_save_acc(original_predictions, 
                      labels, 
                      cleaned_predictions= None, 
                      save_file = None, 
                      write_outputs = False,
                      model_args = None,
                      data_args= None,):


    ## Assert that the lengths of the predictions and labels are the same
    assert len(original_predictions) == len(labels)
    if cleaned_predictions:
        assert len(cleaned_predictions) == len(labels)

    cleaned_correct = 0
    original_correct = 0
    cleaned_length_error =0
    original_length_error =0
    to_write_lines = []
    for i in range(len(original_predictions)):
        original, label = original_predictions[i].lower().strip(), labels[i].lower().strip()
        correctly_predicted = False
        correct_after_clean = False

        if original == label:
            original_correct +=1
            correctly_predicted = True
        if len(original) != len(label):
            original_length_error +=1

        if cleaned_predictions:
            cleaned = cleaned_predictions[i].lower().strip()

            if cleaned == label:
                cleaned_correct +=1
                correctly_predicted = True

                if original != label:
                    correct_after_clean = True
            if len(cleaned) != len(label):
                cleaned_length_error +=1

    
        if write_outputs:
            if correctly_predicted:
                x = emoji.emojize(f'Raw: {original} | Cleaned: {cleaned} | Label: {label} | correct after cleaning: {correct_after_clean} :check_mark_button: \n')
            else:
                x = emoji.emojize(f'Raw: {original} | Cleaned: {cleaned} | Label: {label}    :cross_mark: \n')

            to_write_lines.append(x)
            to_write_lines.append('---------------------------------------------------------------------------------- \n\n')

    cleaned_acc = float (cleaned_correct / len(original_predictions))
    original_acc = float (original_correct / len(original_predictions))
    cleaned_length_error = float ((cleaned_length_error / len(original_predictions) ))
    original_length_error = float ((original_length_error / len(original_predictions) ))
    num_examples = len(original_predictions)
    
    print(f'Number of Examples {num_examples}\n')
    print(f'Cleaned ACCURACY:  { cleaned_acc}\n')
    print(f'Orginal ACCURACY:  { original_acc}\n')
    print(f'Cleaned Length error:  { cleaned_length_error}\n')
    print(f'Original Length error:  { original_length_error}\n')
    print(f'----------------------------------------------------- \n\n')

    if save_file:
        with open(save_file, 'w') as f:
            f.write(f'Model: {model_args.model_name_or_path}\n')
            f.write(f'dataset: {data_args.dataset} - {data_args.split} split\n')
            f.write(f'Prompt: {data_args.prompt_head} \n')
            f.write(f'Number of Shots: {data_args.n_shots} \n')
            
            f.write(f'Number of Examples {num_examples}\n')
            f.write(f'Cleaned ACCURACY:  { cleaned_acc}\n')
            f.write(f'Orginal ACCURACY:  { original_acc}\n')
            f.write(f'Cleaned Length error:  { cleaned_length_error}\n')
            f.write(f'Original Length error:  { original_length_error}\n')
            f.write(f'----------------------------------------------------- \n\n')
            f.writelines(to_write_lines)
