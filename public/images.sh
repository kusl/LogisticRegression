echo '<div style="display:flex;justify-content:center;align-items:center;"><div>' >> index.html

for image in *.png
do
  echo -n '<img src="' >> index.html
  echo -n $image >> index.html
  echo -n '" alt="draw data">' >> index.html
done
echo '</div></div></body></html>' >> index.html