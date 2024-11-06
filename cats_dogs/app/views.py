import torch
from django.utils.decorators import method_decorator
from torchvision import models, transforms
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views import View
from PIL import Image
from torchvision.models import ResNet18_Weights


class ImageClassifier:
    def __init__(self, model_path):
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_classes = 2
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
        except RuntimeError as e:
            print(f"Ошибка загрузки модели: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify_image(self, image_file):
        try:
            image = Image.open(image_file).convert('RGB')
            image = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                output = self.model(image)
                _, predicted = torch.max(output, 1)
                label = 'Сat' if predicted.item() == 0 else 'Dog'
            return label
        except Exception as e:
            raise ValueError("Ошибка обработки изображения: " + str(e))

classifier = ImageClassifier('models/model.pth')

class ClassifyView(View):
    @method_decorator(csrf_exempt)
    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image provided'}, status=400)

        try:
            label = classifier.classify_image(image_file)
            return JsonResponse({'label': label})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

def main(request):
    return render(request, 'site/main.html', {})