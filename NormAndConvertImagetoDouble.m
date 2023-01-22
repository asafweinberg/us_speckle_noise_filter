function NewImage = NormAndConvertImagetoDouble(Im)

NewImage = double(Im) - double(min(Im(:)));
NewImage = NewImage / max(NewImage(:)); 
