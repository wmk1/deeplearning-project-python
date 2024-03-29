def remove_noise(input_text):
    dirty_words = ["is", "a", "on", "i", "and", "or", "to"]
    words = input_text[1].split(" ")
    noise_free_words = [word for word in words if word not in dirty_words]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text

def remove_noise_train(input_text):
    dirty_words = ["is", "a", "on", "i", "and", "or", "to", "ate", "something", "the", "how", "my", "at"]
    words = input_text[2].split(" ")
    noise_free_words = [word for word in words if word not in dirty_words]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text